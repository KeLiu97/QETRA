import string
import torch
import pandas as pd
from rouge import FilesRouge
from tqdm import tqdm
from transformers.models.t5 import T5ForConditionalGeneration, T5Tokenizer
from precision_recall_fscore import compute

def get_title(prefix, input_text):

    input_ids = tokenizer(str(prefix) + ": " + str(input_text) ,return_tensors="pt", max_length=512, padding="max_length", truncation=True)

    summary_text_ids = model.generate(
        input_ids=input_ids["input_ids"].to(DEVICE),
        attention_mask=input_ids["attention_mask"].to(DEVICE),
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        length_penalty=1.2,
        max_length=48,
        min_length=2,
        num_return_sequences= 5,
        num_beams=5,
    )
    title = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)
    if(title[-1] in string.punctuation):
        title = title[:-1] + " " +title[-1]
    return title

if __name__ == '__main__':
    prefix = "Python"
    originalTitle = "Dictionaries of dictionaries merge"
    body = """
       I need to merge multiple dictionaries, here's what I have for instance: 
       With A B C and D being leaves of the tree, like {"info1":"value", "info2":"value2"}
       There is an unknown level(depth) of dictionaries, it could be {2:{"c":{"z":{"y":{C}}}}}
       In my case it represents a directory/files structure with nodes being docs and leaves being files.
       I want to merge them to obtain:
       I'm not sure how I could do that easily with Python.
       """

    code = """
        dict1 = {1:{"a":{A}}, 2:{"b":{B}}}
        dict2 = {2:{"c":{C}}, 3:{"d":{D}}
        dict3 = {1:{"a":{A}}, 2:{"b":{B},"c":{C}}, 3:{"d":{D}}}
       """
    input_text = ' '.join(originalTitle.split()) + " <body> " + ' '.join(body.split()) + " <code> " + ' '.join(code.split())
    title = get_title(prefix, input_text)
    print(title)