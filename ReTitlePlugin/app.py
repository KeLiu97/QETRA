# coding=utf8
import string

import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import time
import numpy as np
import translators as ts
from transformers import RobertaTokenizer, RobertaForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
from flask import send_file

def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 0, keepdims = True)
    s = x_exp / x_sum
    return s.tolist()



def get_title(tokenizer, model, prefix, input_text):
    input_ids = tokenizer(prefix+": "+input_text ,return_tensors="pt", max_length=512, padding="max_length", truncation=True)
    summary_text_ids = model.generate(
        input_ids=input_ids["input_ids"],
        attention_mask=input_ids["attention_mask"],
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        length_penalty=1.2,
        max_length=48,
        min_length=2,
        num_beams=5,
        num_return_sequences=5
    )
    result_list = []
    for i, beam_output in enumerate(summary_text_ids):
        title = tokenizer.decode(beam_output, skip_special_tokens=True)
        if (title[-1] in string.punctuation):
            title = title[:-1] + " " + title[-1]
        result_list.append(title)
    return result_list

app = Flask(__name__,template_folder="page", static_folder="page")
app.config['JSON_AS_ASCII'] = False
app.config['WTF_CSRF_CHECK_DEFAULT'] = False
from flask.json import JSONEncoder as _JSONEncoder

class JSONEncoder(_JSONEncoder):
    def default(self, o):
        import decimal
        if isinstance(o, decimal.Decimal):
            return float(o)
        super(JSONEncoder, self).default(o)
app.json_encoder = JSONEncoder
CORS(app, supports_credentials=True)

@app.route('/so',methods=['POST'])
def so():
    time_start = time.time()
    title = request.values.get('OriginalTitle')
    desc = request.values.get('Desc')
    tag = request.values.get('Tag')
    # desc = ts.alibaba(desc, from_language="zh", to_language="en")

    input_text =  ' '.join(title.split()) + " <body> " + ' '.join(desc.split()) #+ " <code> " + ' '.join(code.split())
    title_list = get_title(gen_tokenizer, gen_model, tag, input_text)
    # title_list = [ts.alibaba(title, from_language="en", to_language="zh") for title in title_list]
    time_end = time.time()
    return jsonify({'title_List':title_list, 'time':round(time_end - time_start,2)})



if __name__ == '__main__':
    gen_model = T5ForConditionalGeneration.from_pretrained("QETRA")
    gen_tokenizer = T5Tokenizer.from_pretrained("QETRA")
    app.run(port=5000)
