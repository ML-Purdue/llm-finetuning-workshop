#!/usr/bin/env python3

from transformers import pipeline
from IPython import embed

if __name__ == '__main__':
    prompt = \
    """Logically conclude the incomplete sentence.
    All men are mortal. Socrates is a man. Therefore, Socrates is mortal.
    All programmers are happy. Alice is a programmer. Therefore, Alice is happy.
    """
    
    user_input = "All elephants are heavy. Pinocchio is a toy. Therefore, Pinocchio is"
    
    model = pipeline(model='google/flan-t5-base', max_new_tokens=20)
    print(model(prompt + user_input))
    embed()
