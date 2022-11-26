# Dependencies
from flask import Flask, request, jsonify
import traceback
import torch
from datasets import Dataset
import transformers
from transformers import (
  AdamW,
  BertConfig,
  BertModel,
  BertTokenizer,
  DistilBertTokenizer,
  DistilBertModel,
  DistilBertForSequenceClassification,
  BertForSequenceClassification)
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import captum
from captum.attr import LayerIntegratedGradients
import json

from flask_cors import CORS

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from IPython.display import display, HTML
import matplotlib as mpl
from matplotlib.colors import Normalize, rgb2hex
import pandas as pd

from app_utils import *

# Your API definition
app = Flask(__name__)
CORS(app)

# global variables
ref_token_id = None
sep_token_id = None
cls_token_id = None

model = None
device = None
tokenizer = None

parameters = {"model": "bhadresh-savani/bert-base-go-emotion",  # model_type: t5-base/t5-large
    "max_source_length": 512,  # max length of source text
    "SEED": 42,
    "out_dir": "./",
    "hidden_size": 768,
    "num_classes": 28}

index_label = {0:"admiration", 1:"amusement", 2:"anger", 3:"annoyance", 4:"approval", 5:"caring", 6:"confusion",
            7:"curiosity", 8:"desire", 9:"disappointment", 10:"disapproval", 11:"disgust", 12:"embarrassment",
            13:"excitement", 14:"fear", 15:"gratitude", 16:"grief", 17:"joy", 18:"love", 19:"nervousness",
            20:"optimism", 21:"pride", 22:"realization", 23:"relief", 24:"remorse", 25:"sadness",
            26:"surprise", 27:"neutral"}
label_list = list(index_label.values())


@app.route('/classify', methods=['POST'])
def classify():
    try:
        json_ = request.json
        print(json_)
        text = json_["text"]

        classes, class_colors, class_words, class_attrs = predict(text, model, tokenizer, device)

        return jsonify({'classes': classes, 'class_colors': class_colors, 'class_words': class_words, "class_attrs": class_attrs})

    except:

        return jsonify({'trace': traceback.format_exc()})

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345



    cuda =  torch.cuda.is_available()
    device = torch.device("cuda") if cuda else torch.device("cpu")
    tokenizer = DistilBertTokenizer.from_pretrained(parameters["model"])
    model = BertForSequenceClassification.from_pretrained(parameters["model"])
    model = model.to(device)
    print ('Model loaded')
    
    ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
    sep_token_id = tokenizer.sep_token_id # A token used as a separator at the end of the text.
    cls_token_id = tokenizer.cls_token_id # A token used for prepending to the word sequence

    app.run(port=port, debug=True)