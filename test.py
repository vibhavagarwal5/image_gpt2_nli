import copy
import json
import logging
import os
import random
import time
from argparse import ArgumentParser
from pprint import pformat

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import *

from dataset import SPECIAL_TOKENS_DICT, InferenceDataset, collate_fn, get_data, ID2LABEL


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint_dir", type=str,
                        default="", help="short name of the model")
    parser.add_argument("--model_checkpoint", type=str,
                        default="", help="name of the model")
    parser.add_argument("--data_path", type=str,
                        default="/home/hdd1/vibhav/VE-SNLI/mycode-vesnli/dataset/e-SNLI-VE")
    parser.add_argument("--data_type", type=str, default="dev")
    parser.add_argument("--save_file_prefix", type=str,
                        default="generated")
    parser.add_argument("--save_output", action="store_true")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available()
                        else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--no_image", action="store_true",
                        help="To process image or not")
    parser.add_argument("--no_premise", action="store_true",
                        help="To process premise or not")
    parser.add_argument("--with_expl", action="store_true",
                        help="To use explanations or not")
    parser.add_argument("--classify", action="store_true",
                        help="To e labels as well for classification head")
    parser.add_argument("--small_data", type=int,
                        default=-1, help='small data size')

    parser.add_argument("--do_sample", action='store_true',
                        help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--beam_search", action='store_true',
                        help="Set to use beam search instead of sampling")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size")
    parser.add_argument("--max_length", type=int, default=40,
                        help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=6,
                        help="Minimum length of the output utterances")
    parser.add_argument("--length_penalty", type=float,
                        default=0.3, help="length penalty")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7,
                        help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()
    return args


def sample_sequence(model, length, context, eos_token_id=None):
    input_ids, image = context
    generated = input_ids
    past = None
    with torch.no_grad():
        for _ in range(length):
            output, past = model(input_ids=generated, images=image)
            next_token_logits = output[0, -1, :]
            next_token = torch.argmax(next_token_logits)
            generated = torch.cat((generated, next_token.view(1, 1)), dim=1)
            if next_token.item() == eos_token_id:
                break
            input_ids = next_token.view(1, 1)
    return generated


def test_loop(model, tokenizer, dataloader, args):
    save_point = os.path.join(args.model_checkpoint_dir,
                              f"{args.save_file_prefix}_{args.data_type}.csv")
    data_to_save = []
    for idx, batch in enumerate(tqdm(dataloader)):
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        if args.no_image:
            if args.classify:
                input_ids, lm_label, label, mc_token_ids, input_mask = batch
            else:
                input_ids, lm_label, label, input_mask = batch
        else:
            if args.classify:
                image, input_ids, lm_label, label, mc_token_ids, input_mask = batch
            else:
                image, input_ids, lm_label, label, input_mask = batch

        if args.classify:
            if args.no_image:
                logits, mc_logits = model(
                    input_ids=input_ids,
                    mc_token_ids=mc_token_ids
                )
            else:
                logits, mc_logits = model(
                    input_ids=input_ids,
                    images=image,
                    mc_token_ids=mc_token_ids
                )
            grnd_label = ID2LABEL[label[0].item()]
            gen_label = ID2LABEL[mc_logits.argmax(dim=1)[0].item()]
            input_ids = input_ids[0].tolist() + tokenizer.encode(
                gen_label,
                [SPECIAL_TOKENS_DICT['additional_special_tokens'][2]])
            input_ids = torch.tensor([input_ids]).long().to(args.device)

        if args.no_image:
            output = model.generate(
                input_ids,
                num_beams=args.beam_size,
                max_length=args.max_length + input_ids.shape[-1],
                min_length=args.min_length,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                do_sample=args.do_sample,
                length_penalty=args.length_penalty,
            )
        else:
            output = sample_sequence(
                model=model,
                context=(input_ids, image),
                length=args.max_length,
                eos_token_id=tokenizer.eos_token_id,
            )

        in_sent = tokenizer.decode(input_ids[0])
        expl = tokenizer.decode(list(filter(lambda x: x != -100, lm_label[0])),
                                skip_special_tokens=True)
        out_expl = tokenizer.decode(output[0, input_ids.shape[-1]:],
                                    skip_special_tokens=True)

        to_save = [in_sent, expl, out_expl]
        columns = ['model_input', 'expl', 'gen_expl']
        if args.classify:
            to_save += [grnd_label, gen_label]
            columns += ['grnd_label', 'gen_label']
        data_to_save.append(to_save)

        if idx % int(len(dataloader) / 50) == 0:
            print('MODEL_INPUT: ', in_sent)
            print('GROUND_EXPL: ', expl)
            print('GEN_EXPL: ', out_expl)
            if args.classify:
                print('GROUND_LBL: ', grnd_label)
                print('GEN_LBL: ', gen_label)
            print('--------------------------------')
            if args.save_output:
                pd.DataFrame(data_to_save, columns=columns).to_csv(
                    save_point, sep='\t', index=None)
    if args.save_output:
        pd.DataFrame(data_to_save, columns=columns).to_csv(
            save_point, sep='\t', index=None)
    bleu_prediction(save_point)
    if args.classify:
        df = pd.read_csv(save_point, sep='\t')
        lbl_accuracy = df[df['grnd_label'] ==
                          df['gen_label']].shape[0] / df.shape[0]
        print('Label Accuracy:', lbl_accuracy)


def bleu_prediction(generated_file):
    candidates = []
    references = []
    df = pd.read_csv(generated_file, sep='\t')
    df.dropna(inplace=True)
    expls = zip(df['gen_expl'].tolist(), df['expl'].tolist())
    for gen, grnd in expls:
        candidates.append(gen.strip().split())
        references.append([grnd.strip().split()])
    bleu_score = 100 * corpus_bleu(references, candidates)
    print('BLEU: ', bleu_score)
    return bleu_score


if __name__ == "__main__":
    set_seed(42)
    args = get_args()
    if 'eSNLI' in args.data_path:
        args.no_image = True
    if not args.no_image:
        args.no_premise = True

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')
    logger.info(f"Arguments: {pformat(args)}")
    logging.info('Loading model params from ' + args.model_checkpoint)
    logger.info(f'Image not used:{args.no_image}')
    logger.info(f'Premise not used:{args.no_premise}')
    logger.info(f'Explanations used:{args.with_expl}')

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint_dir)
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    model_config = GPT2Config.from_pretrained(args.model_checkpoint_dir)
    if args.no_image:
        if args.classify:
            import org_gpt2_291
            model = org_gpt2_291.GPT2DoubleHeadsModel(model_config)
            args.beam_size = 1
        else:
            model = GPT2LMHeadModel(model_config)
    else:
        import image_gpt2_291
        if args.classify:
            model = image_gpt2_291.GPT2DoubleHeadsModel(model_config)
            args.beam_size = 1
        else:
            model = image_gpt2_291.GPT2LMHeadModel(model_config)
    model.load_state_dict(torch.load(os.path.join(args.model_checkpoint_dir,
                                                  args.model_checkpoint)))
    model.to(args.device)
    model.eval()

    logging.info('Loading test data from ' + args.data_path)
    dataset = InferenceDataset(args.data_type, tokenizer, args)
    if args.small_data != -1:
        logger.info('Using small subset of data')
        dataset = Subset(dataset, list(range(args.small_data)))
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=8,
                            collate_fn=lambda x: collate_fn(x, -100, args))

    test_loop(model, tokenizer, dataloader, args)
