#!/usr/bin/env python3

import transformers
from .. import const
from .arch import get_model
from ..dataset import MARCO


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad: trainable_params += param.numel()
    print(f'trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}')


def train():
    tokenizer, model = get_model()
    train, test = [MARCO(const.DATA_DIR, split, tokenizer) for split in ['train', 'test']]

    print_trainable_parameters(model)

    trainer = transformers.Trainer(
            model=model,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=const.BATCH_SIZE,
                gradient_accumulation_steps=const.GRAD_ACC_STEPS,
                warmup_steps=const.WARMUP_STEPS,
                num_train_epochs=const.EPOCHS,
                learning_rate=const.LR,
                fp16=True,
                output_dir=const.SAVE_DIR,
                optim=const.OPTIMIZER),
            train_dataset=train,
            eval_dataset=test)
    model.config.use_cache = False
    trainer.train()
    model.save_pretrained(const.SAVE_DIR)


if __name__ == '__main__':
    train()
