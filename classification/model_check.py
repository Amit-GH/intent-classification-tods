import json
import os
import sys

import numpy as np

import torch
from transformers import DistilBertTokenizer

from data_loader.S3Loader import load_model
from classification.model import load_pretrained_model


def _load_test_examples(params: dict):
    root_path = params.get('root_path', '../../')
    complete_path = os.path.join(root_path, 'intent-classification-tods/data/test_examples.json')
    examples: dict = json.load(open(complete_path, "r"))
    return examples


def evaluate_model():
    model, tokenizer = load_pretrained_model(model_directory="../saved_models/fine_tuned_cfn")

    print('All labels: {}'.format(model.config.label2id.keys()))
    examples = _load_test_examples({})
    for label, sentences in examples.items():
        for sent in sentences:
            sentence_tokenize = tokenizer(sent, return_tensors="pt", padding=True)
            res = model(**sentence_tokenize).logits
            probs = torch.softmax(res, dim=1).tolist()[0]
            pred = int(np.argmax(probs))
            if pred != model.config.label2id[label]:
                print("sent={}, exp={}, actual={}".format(sent, label, model.config.id2label[pred]))


if __name__ == '__main__':
    evaluate_model()
    sys.exit()
