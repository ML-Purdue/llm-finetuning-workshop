#!/usr/bin/env python3

from src.model.arch import get_model
from peft import PeftModel
from src import const
import torch


class SummarizerModel:
    def __init__(self):
        self.tokenizer, model = get_model()
        self.model = PeftModel.from_pretrained(model, const.SAVE_DIR).to(const.DEVICE)

    def predict(self, task):
        with torch.no_grad():
            return self.model(self.tokenizer(task, **const.TOKENIZER_ARGS).input_ids.to(const.DEVICE))


if __name__ == '__main__':
    model = SummarizerModel()
    while True: model.predict(input('Enter a task as input: '))
