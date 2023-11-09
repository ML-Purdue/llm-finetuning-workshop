#!/usr/bin/env python3

from torch.utils.data import Dataset
from src import const
import json


class MARCO(Dataset):
    def __init__(self, directory, split, tokenizer):
        self.data = []
        data = json.load(open(f'{directory}/law.{split}.json'))['data']
        for paragraph in data['data']:
            for sample in paragraph['paragraphs']:
                context = sample['context']
                for question in sample['qas']:
                    for answer in question['answers']:
                        self.data.append({'context': context, 
                                          'question': question['question'], 
                                          'answer': answer['text']})

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        context = self.tokenizer(f"{sample['context']} {sample['question']}", **const.TOKENIZER_ARGS)
        answer = self.tokenizer(sample['answer'], **const.TOKENIZER_ARGS)
        return {'input_ids': context.input_ids[0],
                'attention_mask': context.attention_mask[0],
                'labels': answer.input_ids[0]}
