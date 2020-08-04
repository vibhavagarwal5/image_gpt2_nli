import json
import logging
import os
import pickle
from itertools import chain
from pprint import pformat

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset

LABEL_TOKENS_DICT = {
    'contradiction': 0,
    'neutral': 1,
    'entailment': 2
}
SPECIAL_TOKENS_DICT = {
    'eos_token': '<eos>',
    'additional_special_tokens': ['<premise>', '<hypothesis>', '<expl>']
}


def get_data(data_type, args):
    data = {}
    data['expl'] = [line.rstrip() for line in open(
        os.path.join(args.data_path, f"expl_1.{data_type}"), 'r')]
    data['label'] = [line.rstrip() for line in open(
        os.path.join(args.data_path, f"labels.{data_type}"), 'r')]
    data['label_int'] = [
        LABEL_TOKENS_DICT[i] for i in data['label']]
    data['hypothesis'] = [line.rstrip() for line in open(
        os.path.join(args.data_path, f"s2.{data_type}"), 'r')]
    if args.no_image:
        data['premise'] = [line.rstrip() for line in open(
            os.path.join(args.data_path, f"s1.{data_type}"), 'r')]
    else:
        data['image_f'] = [line.rstrip() for line in open(
            os.path.join(args.data_path, f"images.{data_type}"), 'r')]
    return data


class InferenceDataset(Dataset):
    def __init__(self, data_type, tokenizer, args):
        self.data = get_data(data_type, args)
        self.tokenizer = tokenizer
        self.args = args
        if not self.args.no_image:
            self.all_images_np = np.load(
                '/home/hdd1/vibhav/VE-SNLI/e-SNLI-VE/data/flickr30k_resnet101_bottom_up_img_features.npy')
            f = open(
                '/home/hdd1/vibhav/VE-SNLI/e-SNLI-VE/data/filenames_77512.json', 'r')
            self.all_image_names = json.load(f)

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, index):
        add_spl_tkn = SPECIAL_TOKENS_DICT['additional_special_tokens']

        hypothesis = self.data['hypothesis'][index]
        hypothesis = add_spl_tkn[1] + hypothesis + add_spl_tkn[2]
        if self.args.no_premise:
            input_ids = self.tokenizer.encode(hypothesis)
        else:
            premise = self.data['premise'][index]
            premise = add_spl_tkn[0] + premise
            input_ids = self.tokenizer.encode(premise, hypothesis)
        expl = self.data['expl'][index] + self.tokenizer.eos_token
        expl_ids = self.tokenizer.encode(expl)
        if self.args.with_expl:
            input_ids = input_ids + expl_ids
        input_ids = torch.tensor(input_ids).long()
        output = (input_ids,)
        if self.args.no_image:
            lm_label = [-100] * (len(input_ids) - len(expl_ids)) + expl_ids
        else:
            lm_label = [-100] * \
                (len(input_ids) - len(expl_ids) + 36) + expl_ids
        lm_label = torch.tensor(lm_label).long()
        output = output + (lm_label,)

        label = torch.tensor(self.data['label_int'][index]).long()
        output = output + (label,)

        if not self.args.no_image:
            image = self.all_images_np[self.all_image_names.index(
                self.data['image_f'][index])]
            output = output + (image,)
        return output   # input_ids, lm_label, label, image


def collate_fn(batch, pad_token, args, no_pad=False):
    def padding(seq, max_len, pad_token):
        padded_mask = torch.ones((len(seq), max_len)).long() * pad_token
        for i in range(len(seq)):
            padded_mask[i, :len(seq[i])] = seq[i]
        return padded_mask

    input_ids, lm_label, label = [], [], []
    if not args.no_image:
        image = []
    for i in batch:
        input_ids.append(i[0])
        lm_label.append(i[1])
        label.append(i[2])
        if not args.no_image:
            image.append(i[3])

    max_len_inp_ids = max(len(s) for s in input_ids)
    max_len_lm_label = max(len(s) for s in lm_label)
    input_ids = padding(input_ids, max_len_inp_ids, pad_token)
    lm_label = padding(lm_label, max_len_lm_label, pad_token)

    label = torch.tensor(label).long()
    output = (input_ids, lm_label, label)
    if not args.no_image:
        image = torch.tensor(image)
        input_mask = input_ids.ne(pad_token).long()
        image_mask = torch.ones((len(image), 36)).long()
        input_mask = torch.cat([image_mask, input_mask], dim=1)
        output = (image,) + output + (input_mask,)
    else:
        input_mask = input_ids.ne(pad_token).long()
        output = output + (input_mask,)

    return output   # image, input_ids, lm_label, label, input_mask


'''main'''
if __name__ == "__main__":
    from transformers import *
    from torch.utils.data import DataLoader
    import itertools
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data_type", type=str,
                        default="dev", help="dev or train or test")
    parser.add_argument("--data_path", type=str,
                        default="/home/hdd1/vibhav/VE-SNLI/mycode-vesnli/dataset/e-SNLI-VE", help="Path of the dataset")
    parser.add_argument("--no_image", action="store_true",
                        help="To process image or not")
    parser.add_argument("--no_premise", action="store_true",
                        help="To process premise or not")
    parser.add_argument("--with_expl", action="store_true",
                        help="To use explanations or not")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    if 'e-SNLI-VE' in args.data_path:
        args.no_image = False
    else:
        args.no_image = True
    if not args.no_image:
        args.no_premise = True
    print(f"Arguments: {pformat(args)}")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)

    dataset = InferenceDataset(args.data_type, tokenizer, args)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            collate_fn=lambda x: collate_fn(x, tokenizer.eos_token_id, args))

    batch = next(iter(dataloader))
    if args.no_image:
        input_ids, lm_label, label, input_mask = batch
    else:
        image, input_ids, lm_label, label, input_mask = batch

    for i, v in enumerate(batch):
        print(i, v.shape)
    print('input_ids', input_ids[0])
    print('input_ids', tokenizer.convert_ids_to_tokens(input_ids[0]))
    print('lm_label', lm_label[0])
    print('lm_label', tokenizer.convert_ids_to_tokens(lm_label[0]))
    print('input_mask', input_mask[0])
    print('label', label)
