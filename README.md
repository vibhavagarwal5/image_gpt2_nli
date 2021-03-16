# Image GPT (Quick prototyping)

Finetuned GPT2 by inputting Faster RCNN object features concatenated with the textual tokens. Tested with [e-SNLI-VE dataset](https://github.com/virginie-do/e-SNLI-VE).

**Result:** No improvement in terms of generation by adding or not adding image features to the textual tokens. Since there is no image based pretrain objective, adding image input to the model doesnt make a difference. 
