#!/usr/bin/env python3

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'

BASE_MODEL = 'bigscience/bloom-1b7'
DEVICE = 'cuda'

# Trainer Arguments
LR = 1e-3
EPOCHS = 3
BATCH_SIZE = 12
WARMUP_STEPS = 2
GRAD_ACC_STEPS = 4
SAVE_DIR = BASE_DIR / 'model'
OPTIMIZER = 'paged_adamw_8bit'
CHECKPOINT = SAVE_DIR / 'checkpoint-19500'

# LoRA Config
TARGET_MODULES = ['query_key_value']

TOKENIZER_ARGS = {'return_tensors': 'pt',
                  'padding': 'max_length',
                  'truncation': True,
                  'max_length': 3072}
