import json
import os
import sys

import numpy as np

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertModel, AutoModelForSequenceClassification

from data_loader.DataLoader import load_mapped_data, ClincDataSet, Group, load_model_from_disk, load_object_from_disk
from data_loader.S3Loader import load_model
from classification.model import load_pretrained_model, calculate_class_weights, perform_validation, device, WandbMode


def _load_test_examples(params: dict):
    root_path = params.get('root_path', '../../')
    complete_path = os.path.join(root_path, 'intent-classification-tods/data/test_examples.json')
    examples: dict = json.load(open(complete_path, "r"))
    return examples


def evaluate_model(save_path=None):
    if save_path:
        id2label = load_object_from_disk("../saved_models/class_imbalance/id2label.pickle")
        label2id = load_object_from_disk("../saved_models/class_imbalance/label2id.pickle")
        empty_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id
        )
        model = load_model_from_disk(save_path, empty_model)
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    else:
        model, tokenizer = load_pretrained_model(model_directory="../saved_models/multiclass_cfn")

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


def check_label_wise_accuracy(save_path=None, model_directory=None):
    if save_path:
        id2label = load_object_from_disk("../saved_models/class_imbalance/id2label.pickle")
        label2id = load_object_from_disk("../saved_models/class_imbalance/label2id.pickle")
        empty_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id
        )
        model = load_model_from_disk(save_path, empty_model)
        model.to(device)
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    elif model_directory:
        model, tokenizer = load_pretrained_model(model_directory=model_directory)
        model.to(device)
        id2label = None
    else:
        raise Exception('provide model path to load.')

    mapped_data = load_mapped_data({}, balance_split=False)
    val_dataset = ClincDataSet(mapped_data[Group.val.value][1], tokenizer)
    val_class_weights = calculate_class_weights(mapped_data[Group.val.value][0])
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)
    if id2label:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(val_class_weights, dtype=torch.float32, device=device))
    else:
        criterion = nn.CrossEntropyLoss()
    _, _ = perform_validation(model, val_loader, criterion, log_metric=True)
    print('done')


if __name__ == '__main__':
    # Initialize but disable wandb for checking code.
    os.environ.setdefault('WANDB_API_KEY', '713a778aae8db6219a582a6b794204a5af2cb75d')
    wandb.init(project="ms-project-701", entity="amitgh", config={}, mode=WandbMode.DISABLED.value)

    # evaluate_model(save_path="../saved_models/class_imbalance/dict_values([0.0001, 15000, 10, 512, 5, 'DistilBertModel+Linear', 'modified-CLINC150'])")
    # evaluate_model()

    check_label_wise_accuracy(
        save_path="../saved_models/class_imbalance/dict_values([0.0001, 15000, 10, 512, 5, 'DistilBertModel+Linear', 'modified-CLINC150'])"
    )
    # check_label_wise_accuracy(
    #     model_directory="../saved_models/multiclass_cfn"
    # )
    # check_label_wise_accuracy(
    #     model_directory="../saved_models/fine_tuned_cfn"
    # )
    sys.exit()
