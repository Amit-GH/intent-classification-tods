import json
import os
import pickle
import sys

import numpy as np

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertModel, AutoModelForSequenceClassification

from data_loader.DataLoader import load_mapped_data, ClincDataSet, Group, load_model_from_disk, load_object_from_disk
from data_loader.S3Loader import load_model, load_model_from_single_file, s3_fileobj
from classification.model import load_pretrained_model, calculate_class_weights, perform_validation, device, WandbMode


def _load_test_examples(params: dict):
    root_path = params.get('root_path', '../../')
    complete_path = os.path.join(root_path, 'intent-classification-tods/data/test_examples.json')
    examples: dict = json.load(open(complete_path, "r"))
    return examples


def load_pretrained_model_tokenizer(s3_params=None, model_name=None, save_path=None, model_directory=None):
    """
    Use s3_params and model_name for loading single file model from S3.
    Use save_path for loading single file model from local disk.
    Use model_directory for loading 2 file model from local disk.
    Returns model, tokenizer, and boolean flag telling if model was trained on balanced data or not.
    """
    if s3_params:
        bucket = s3_params['bucket']
        path_to_model = s3_params['path_to_model']
        model = load_model_from_single_file(
            s3_params=s3_params,
            model_name=model_name
        )
        model.to(device)
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        with s3_fileobj(bucket, f'{path_to_model}/id2label.pickle') as f:
            id2label = pickle.loads(f.read())
    elif save_path:
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

    # We use id2label variable to determine if the model was trained on balanced or unbalanced data. Not a good way but
    # fine for now.
    return model, tokenizer, id2label is None


def evaluate_model(s3_params=None, model_name=None, save_path=None, model_directory=None):
    model, tokenizer, trained_on_balanced_data = load_pretrained_model_tokenizer(
        s3_params, model_name, save_path, model_directory
    )

    print('All labels: {}'.format(model.config.label2id.keys()))
    examples = _load_test_examples({})
    for label, sentences in examples.items():
        for sent in sentences:
            sentence_tokenize = tokenizer(sent, return_tensors="pt", padding=True)
            for k, v in sentence_tokenize.items():
                sentence_tokenize[k] = v.to(device)
            res = model(**sentence_tokenize).logits
            probs = torch.softmax(res, dim=1).tolist()[0]
            pred = int(np.argmax(probs))
            if pred != model.config.label2id[label]:
                print("sent={}, exp={}, actual={}".format(sent, label, model.config.id2label[pred]))


def check_label_wise_accuracy(s3_params=None, model_name=None, save_path=None, model_directory=None, group=Group.test):
    """
    Use s3_params and model_name for loading single file model from S3.
    Use save_path for loading single file model from local disk.
    Use model_directory for loading 2 file model from local disk.
    :return: The trained model which was already saved.
    """
    model, tokenizer, trained_on_balanced_data = load_pretrained_model_tokenizer(
        s3_params, model_name, save_path, model_directory
    )
    print(f'We will find accuracy on {group.value} dataset.')
    mapped_data = load_mapped_data({}, balance_split=False)
    my_dataset = ClincDataSet(mapped_data[group.value][1], tokenizer)
    print(f'Dataset size: {len(my_dataset)}')
    my_class_weights = calculate_class_weights(mapped_data[group.value][0])
    data_loader = DataLoader(my_dataset, batch_size=512, shuffle=True)

    if not trained_on_balanced_data:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(my_class_weights, dtype=torch.float32, device=device))
    else:
        criterion = nn.CrossEntropyLoss()
    _, _ = perform_validation(model, data_loader, criterion, log_metric=True)
    print('done')


if __name__ == '__main__':
    # Credentials needed if running in some random EC2 instance.
    os.environ.setdefault('AWS_ACCESS_KEY_ID', 'AKIA3OIIZGHY535S56XW')
    os.environ.setdefault('AWS_SECRET_ACCESS_KEY', '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')

    # Initialize but disable wandb for checking code.
    os.environ.setdefault('WANDB_API_KEY', '713a778aae8db6219a582a6b794204a5af2cb75d')
    wandb.init(project="ms-project-701", entity="amitgh", config={}, mode=WandbMode.DISABLED.value)

    # evaluate_model(save_path="../saved_models/class_imbalance/dict_values([0.0001, 15000, 10, 512, 5, 'DistilBertModel+Linear', 'modified-CLINC150'])")
    # evaluate_model(
    #     # s3_params={
    #     #     'bucket': "umass-alexaprize-model-hosting",
    #     #     'path_to_model': "weighted_multiclass_intent_cfn"
    #     # },
    #     # model_name="weighted_class_model.pt",
    #
    #     # model_directory="../saved_models/multiclass_cfn",
    #
    #     model_directory="../saved_models/fine_tuned_cfn"
    # )

    # check_label_wise_accuracy(
    #     save_path="../saved_models/class_imbalance/dict_values([0.0001, 15000, 10, 512, 5, 'DistilBertModel+Linear', 'modified-CLINC150'])"
    # )

    # For checking accuracy on the single file models that are trained with weighed loss function.
    check_label_wise_accuracy(
        s3_params={
            'bucket': "umass-alexaprize-model-hosting",
            'path_to_model': "weighted_multiclass_intent_cfn"
        },
        model_name="weighted_class_model.pt",
        group=Group.test
    )

    # For checking accuracy for 2 file models that are trained with balanced data.
    # check_label_wise_accuracy(
    #     model_directory="../saved_models/multiclass_cfn",
    #     group=Group.test
    # )

    # check_label_wise_accuracy(
    #     model_directory="../saved_models/fine_tuned_cfn",
    #     group=Group.test
    # )
    sys.exit()
