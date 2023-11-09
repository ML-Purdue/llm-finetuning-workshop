#!/usr/bin/env python3

from peft import PeftMOdel
from src import const

import torch
from .. import const
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def get_model():
    bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(const.BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
            const.BASE_MODEL,
            quantization_config=bnb,
            trust_remote_code=True,
            device_map={'': 0})
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=const.TARGET_MODULES,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM')

    return tokenizer, get_peft_model(model, lora)

