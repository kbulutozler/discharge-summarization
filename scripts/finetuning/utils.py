import pandas as pd
import json
import os
from datasets import Dataset, DatasetDict



def tokenize_function(tokenizer, examples, text_column):
    return tokenizer(examples[text_column], padding="max_length")
