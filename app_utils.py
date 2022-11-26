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

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from IPython.display import display, HTML
import matplotlib as mpl
from matplotlib.colors import Normalize, rgb2hex
import pandas as pd

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

# global variables
ref_token_id = None
sep_token_id = None
cls_token_id = None

def construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id, device, tokenizer):
    text_ids = tokenizer.encode(text, add_special_tokens=False)

    # construct input token ids
    input_ids = [cls_token_id] + text_ids + [sep_token_id]

    # construct reference token ids 
    ref_input_ids = [cls_token_id] + [ref_token_id] * (len(input_ids)-2) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(input_ids)

def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)

def predict_(model, inputs, attention_mask=None):
    output = model(inputs, attention_mask=attention_mask)
    output = output["logits"]
    output = torch.sigmoid(output)
    return output

def forward_func(inputs, i, device, model, attention_mask=None):
    pred = predict_(model, inputs,
                   attention_mask=attention_mask)
    #return pred.max(1).values
    pred = torch.index_select(pred, 1, torch.tensor([i], device=device))
    return pred

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

def colorize(attrs, cmap='PiYG'):

    cmap_bound = max([abs(attr) for attr in attrs])

    norm = Normalize(vmin=-cmap_bound, vmax=cmap_bound)

    cmap = mpl.cm.get_cmap(cmap)
    colors = list(map(lambda x: rgb2hex(cmap(norm(x))), attrs))

    return colors

def  hlstr(string, color='white'):
    return f"<mark style=background-color:{color}>{string} </mark>"

def color(word_scores, words):
    colors = colorize(word_scores)
    colored_input = []
    lis = list(map(hlstr, words, colors))
    #print("".join(lis))
    #display(HTML("".join(lis)))
    return lis

def explainability(data, limit, model, device):
    lig = LayerIntegratedGradients(forward_func, model.bert.embeddings)  # embeddings is the first layer
    attributions = []
    predictions = []
    for j,sample in enumerate(data[0:limit]):
        pred = predict_(model, sample[0], attention_mask=sample[2])
        classes_, _ = get_predictions([pred])
        classes_ = classes_[0]

        classes = []
        for i in range(0,28):
            if not classes_[i]: 
                classes.append([])
                continue
            attribution, delta = lig.attribute(inputs=sample[0],
                                          baselines=sample[1],
                                          additional_forward_args=(i, device, model, sample[2]),
                                          return_convergence_delta=True)
            attribution = summarize_attributions(attribution).detach().cpu().numpy()
            attribution = [(attr,k) for k,attr in enumerate(attribution)]
            attribution.sort(key=(lambda x: x[0]), reverse=True)
            classes.append(attribution)
        attributions.append(classes)
        predictions.append(pred)
    return attributions, predictions

def get_predictions(preds):
    all_classes = []
    all_preds = []

    for pred in preds:
        classes = []
        for thresh in [0.32,0.28,0.24,0.20,0.16,0.12]:
            classes = pred.detach().cpu().numpy()[0] > thresh
            flag = False
            for class_ in classes:
                if class_:
                    flag = True
                    break 
            if flag: break
        all_classes.append(classes)
        all_preds.append(list(pred.detach().cpu().numpy()[0]))
    
    return all_classes, all_preds


def predict(text, model, tokenizer, device):
    # cleaning data so as to ensure data is in string type
    source_text = " ".join(text.split())

    ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
    sep_token_id = tokenizer.sep_token_id # A token used as a separator at the end of the text.
    cls_token_id = tokenizer.cls_token_id # A token used for prepending to the word sequence

    data = []
    for i,text in enumerate([source_text]):
        input_ids, ref_input_ids, sep_id = construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id, device, tokenizer)
        attention_mask = construct_attention_mask(input_ids)

        indices = input_ids[0].detach().tolist()
        all_tokens = tokenizer.convert_ids_to_tokens(indices)

        data.append((input_ids, ref_input_ids, attention_mask, all_tokens))

    attributions, predictions = explainability(data, 16, model, device)
    predictions, probabilities = get_predictions(predictions)

    # compute colorful words
    pred = predictions[0]
    probab = probabilities[0]

    classes_ = [index_label[index] for index,i in enumerate(pred) if i]
    print("predicted classes are {}".format(classes_))
    print("Input text is {}".format(source_text))
    print("Important words are:")
    classes = [index for index,i in enumerate(pred) if i]
    class_colors = []
    class_words = []
    for index in classes:
        class_attr = attributions[0][index]
        word_scores = [0 for _ in range(len(class_attr))]
        words = ["" for _ in range(len(class_attr))]
        for m in range(len(class_attr)):
            word_scores[class_attr[m][1]] = class_attr[m][0]
            words[class_attr[m][1]] = data[0][-1][class_attr[m][1]]
            
        print(" ".join(words[1:-1]))
        class_colors.append(color(word_scores[1:-1], words[1:-1]))
        class_words.append(words[1:-1])
    
    class_attrs = color(probab, label_list)

    return classes_, class_colors, class_words, class_attrs

if __name__ == "__main__":

    cuda =  torch.cuda.is_available()
    device = torch.device("cuda") if cuda else torch.device("cpu")
    tokenizer = DistilBertTokenizer.from_pretrained(parameters["model"])
    ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
    sep_token_id = tokenizer.sep_token_id # A token used as a separator at the end of the text.
    cls_token_id = tokenizer.cls_token_id # A token used for prepending to the word sequence
    
    model = BertForSequenceClassification.from_pretrained(parameters["model"])
    model = model.to(device)
    predict("This is really helpful to point out!!", model, tokenizer, device)