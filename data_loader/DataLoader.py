"""
We create a data loader for CLINC150 dataset.
"""
import json
import os
import pickle
import sys
from enum import Enum

import torch
from numpy.random import default_rng
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import DistilBertTokenizer, AutoModelForSequenceClassification, PreTrainedModel, PreTrainedTokenizer

from data_loader.S3Loader import load_model_from_single_file, s3_fileobj, load_model, load_model_from_disk
from data_loader.device_setup import device

rng = default_rng()


class Group(Enum):
    train = 'train'
    val = 'val'
    test = 'test'
    train_oos = 'train_oos'
    val_oos = 'val_oos'
    test_oos = 'test_oos'


class ClincSingleData:
    def __init__(self, sentence: str, label_id: int, group: Group):
        """
        group tells
        :type sentence: object
        """
        self.sentence = sentence
        self.group = group
        self.label_id = label_id


def load_examples(raw_data: list, group: Group, label_to_id: dict) -> (dict, list[ClincSingleData]):
    """
    :param raw_data: List of data for a particular group in the raw format.
    :param group: The group to which all this data belongs to.
    :param label_to_id: Maps string labels to int ids. If label not present here, skip it.
    :return: A tuple with dict and list. Dict has the label counts and list has
        all the examples in Clinc objects.
    """
    data = []
    label_count = {}
    for example in raw_data:
        sentence = example[0]
        if example[1] in label_to_id:
            label_id = label_to_id[example[1]]
            clinc_data = ClincSingleData(sentence, label_id, group)
            data.append(clinc_data)
            label_count[label_id] = label_count.get(label_id, 0) + 1

    return label_count, data


def _load_mapped_examples(raw_data: list, group: Group, custom_label_to_id: dict, clinc_to_custom: dict) -> (
        dict, list[ClincSingleData]):
    """
    Loads raw data into list clinc object and get label count too.
    :param raw_data: the raw clinc data obtained after reading from file.
    :param group: the group that the data belongs to (train, val, etc.).
    :param custom_label_to_id: mapping from custom label string to integer id. It has all possible custom labels.
    :param clinc_to_custom: mapping from string clinc class to custom clinc class. Unmapped clinc classes should be
        mapped to undefined custom class by default.
    :return: a tuple of dict and list.
    """
    data = []
    label_count = {}
    for example in raw_data:
        sentence = example[0]
        clinc_label = example[1]
        custom_label = clinc_to_custom.get(clinc_label, "undefined")
        custom_label_id = custom_label_to_id[custom_label]
        clinc_single_data = ClincSingleData(sentence, custom_label_id, group)
        data.append(clinc_single_data)
        label_count[custom_label_id] = label_count.get(custom_label_id, 0) + 1

    return label_count, data


def print_partial_data(params: dict, count_per_label=5, clinc_labels=None):
    """
    Prints partial data from the original Clinc150 dataset for verification.
    :param params: Has the root_path as a key.
    :param count_per_label: number of items you want to see per label.
    :param clinc_labels: list of labels that you want to see.
    :return: None
    """
    root_path = params.get('root_path', '../../')
    clinc_data_file_path = os.path.join(root_path, 'intent-classification-tods/data/data_full.json')
    clinc_data = json.load(open(clinc_data_file_path, "r"))
    data_raw = clinc_data[Group.train.value]
    label_examples = {}

    for example in data_raw:
        sentence = example[0]
        clinc_label = example[1]
        if clinc_labels and clinc_label not in clinc_labels:
            continue
        label_examples.setdefault(clinc_label, [])
        current_examples = label_examples[clinc_label]
        if len(current_examples) < count_per_label:
            current_examples.append(sentence)

    for label, sentences in label_examples.items():
        print(label)
        for sentence in sentences:
            print('\t' + sentence)


def load_amt_test_data(file_path: str):
    id2label = load_object_from_disk("../saved_models/class_imbalance/id2label.pickle")
    label2id = load_object_from_disk("../saved_models/class_imbalance/label2id.pickle")
    amt_data = json.load(open(file_path, "r"))
    amt_to_custom = dict()
    for label in label2id:
        amt_to_custom[label] = label
    label_count_test, test_data = _load_custom_data(amt_data, amt_to_custom, id2label, label2id, Group.test)
    return test_data


def load_mapped_data(params: dict, balance_split=True, balance_strategy=None, wikihow_data_json=None):
    """
    To load clinc data into training, validation and testing which is mapped into custom classes as defined in
    the file intent_mapping.json.
    balance_split: bool: if true, we balance the number of examples of each class to be in the same range for all
        the three splits.
    balance_strategy: str: can be up or down indicating complete upsampling or complete downsampling.
    :return:
    """
    root_path = params.get('root_path', '../../')
    intent_mapping_file_path = os.path.join(root_path, 'intent-classification-tods/data/intent_mapping.json')
    intent_mapping: dict = json.load(open(intent_mapping_file_path, "r"))

    # A dict mapping clinc class to one unique custom class.
    clinc_to_custom = {}
    for custom_class, clinc_class_list in intent_mapping.items():
        for clinc_class in clinc_class_list:
            assert clinc_class not in clinc_to_custom, "{} cannot be mapped to {}. It's already mapped to {}".format(
                clinc_class, custom_class, clinc_to_custom[clinc_class]
            )
            clinc_to_custom[clinc_class] = custom_class
    print('clinc_to_custom: {}'.format(clinc_to_custom))

    # Load entire clinc data.
    clinc_data_file_path = os.path.join(root_path, 'intent-classification-tods/data/data_full.json')
    clinc_data = json.load(open(clinc_data_file_path, "r"))

    # Create label ids for labels we care about. These are custom labels.
    # At this moment, leave out labels that don't have data except 'undefined' custom label.
    label2id = {}
    id2label = {}
    idx = 0
    for custom_class, clinc_class_list in intent_mapping.items():
        if len(clinc_class_list) > 0 or custom_class == "undefined":
            label2id[custom_class] = idx
            id2label[idx] = custom_class
            idx += 1
    if wikihow_data_json is not None:
        label2id["wikihow"] = idx
        id2label[idx] = "wikihow"
    print('Total custom labels: {}\nCustom label names: {}'.format(len(label2id), label2id.keys()))

    # Load validation, train, test data.
    lc_val, val_data = _load_custom_data(clinc_data, clinc_to_custom, id2label, label2id, Group.val)
    lc_train, train_data = _load_custom_data(clinc_data, clinc_to_custom, id2label, label2id, Group.train)
    # print(sorted(lc_train.items(), key=lambda x: x[0]))
    lc_test, test_data = _load_custom_data(clinc_data, clinc_to_custom, id2label, label2id, Group.test)

    if wikihow_data_json is not None:
        wikihow_data = json.load(open(wikihow_data_json, "r"))
        lc_val_whow, val_data_whow = _load_custom_data(wikihow_data, {"wikihow": "wikihow"}, id2label, label2id,
                                                       Group.val)
        for k, v in lc_val_whow.items():
            lc_val[k] = v
        val_data += val_data_whow

        lc_train_whow, train_data_whow = _load_custom_data(wikihow_data, {"wikihow": "wikihow"}, id2label, label2id,
                                                       Group.train)
        for k, v in lc_train_whow.items():
            lc_train[k] = v
        train_data += train_data_whow

        lc_test_whow, test_data_whow = _load_custom_data(wikihow_data, {"wikihow": "wikihow"}, id2label, label2id,
                                                       Group.test)
        for k, v in lc_test_whow.items():
            lc_test[k] = v
        test_data += test_data_whow

    if balance_split:
        if balance_strategy == "up":
            min_train_count = max_train_count = max(lc_train.values())
            min_val_count = max_val_count = max(lc_val.values())
            min_test_count = max_test_count = max(lc_test.values())
        elif balance_strategy == "down":
            min_train_count = max_train_count = min(lc_train.values())
            min_val_count = max_val_count = min(lc_val.values())
            min_test_count = max_test_count = min(lc_test.values())
        elif balance_strategy is None:
            min_train_count, max_train_count = 1000, 5000
            min_val_count, max_val_count = 400, 5000
            min_test_count, max_test_count = 600, 1500
        else:
            raise Exception("Unsupported balance strategy passed.")

        lc_train_bal, train_data_bal = balance_data(lc_train, train_data, min_train_count, max_train_count)
        lc_val_bal, val_data_bal = balance_data(lc_val, val_data, min_val_count, max_val_count)
        lc_test_bal, test_data_bal = balance_data(lc_test, test_data, min_test_count, max_test_count)
        return {
            Group.train.value: (lc_train_bal, train_data_bal),
            Group.val.value: (lc_val_bal, val_data_bal),
            Group.test.value: (lc_test_bal, test_data_bal),
            "label2id": label2id,
            "id2label": id2label
        }
    else:
        return {
            Group.train.value: (lc_train, train_data),
            Group.val.value: (lc_val, val_data),
            Group.test.value: (lc_test, test_data),
            "label2id": label2id,
            "id2label": id2label
        }


def _load_custom_data(clinc_data: dict, clinc_to_custom: dict, id2label: dict, label2id: dict, group: Group, verbose=False):
    data_raw = clinc_data[group.value]
    label_count, data = _load_mapped_examples(data_raw, group, label2id, clinc_to_custom)
    if verbose:
        print('{} data loaded. No of classes: {}'.format(group.value, len(label_count)))
        print('Count per class:')
        for custom_id, count in sorted(label_count.items(), key=lambda x: x[0]):
            print("{}, id: {}, count: {}".format(id2label[custom_id], custom_id, count))
        print()
    # Shuffle data to maintain randomness even if it is sliced later on.
    rng.shuffle(data)
    return label_count, data


def balance_data(
        label_count: dict,
        data: list[ClincSingleData],
        min_no: int,
        max_no: int
) -> (dict, list[ClincSingleData]):
    """
    Removes class imbalance by either repeating data or pruning data depending on original frequency.
    The final count per class is between min_no and max_no.
    :param max_no: max number of examples needed per classes
    :param min_no: min number of examples needed per class
    :param label_count: original class count statistics. key=id, value=count
    :param data: list of Clinc data objects, shuffled.
    :return: tuple with new label_count and data list.
    """
    id2datalist = {}
    for key in label_count.keys():
        id2datalist[key] = []
    for clinc_data in data:
        id2datalist[clinc_data.label_id].append(clinc_data)

    for _, datalist in id2datalist.items():
        length = len(datalist)
        if length < min_no:
            while len(datalist) < min_no:
                diff = min_no - len(datalist)
                datalist += datalist[:diff]
        elif length <= max_no:
            continue
        else:
            while len(datalist) > max_no:
                datalist.pop()

    label_count = {}
    data = []
    for label_id, datalist in id2datalist.items():
        label_count[label_id] = len(datalist)
        data += datalist

    rng.shuffle(data)
    return label_count, data


def load_data(params: dict) -> dict:
    root_path = params.get('root_path', '../../')
    file_path = os.path.join(root_path, 'intent-classification-tods/data/data_full.json')
    data_full = json.load(open(file_path, "r"))
    data_domains = json.load(open(os.path.join(root_path, 'intent-classification-tods/data/domains.json')))

    # label_to_id = {}
    # label_id = 0
    # for _, v in data_domains.items():
    #     for label in v:
    #         label_to_id[label] = label_id
    #         label_id += 1
    #
    # id_to_label = {}
    # for label, idx in label_to_id.items():
    #     id_to_label[idx] = label

    label_to_id = {"recipe": 0, "balance": 1}
    id_to_label = {0: "recipe", 1: "balance"}

    print("total label count: {}".format(len(label_to_id)))

    combined_data = {
        'label_to_id': label_to_id,
        'id_to_label': id_to_label
    }

    for group in Group:
        if group == Group.val or group == Group.train:  # testing with less data for now.
            data_partial = data_full[group.value]
            label_count, data = load_examples(data_partial, group, label_to_id)
            combined_data[group.value] = (label_count, data)

            # print_partial_data(data)

    return combined_data


class ClincDataSet(Dataset):
    def __init__(self, data: list[ClincSingleData], tokenizer):
        sentences = [clinc.sentence for clinc in data]
        self.labels = [clinc.label_id for clinc in data]
        self.encodings = tokenizer(sentences, truncation=True, padding=True)

    def __getitem__(self, index) -> T_co:
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[index])
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    # load_data({})
    # mapped_data = load_mapped_data({})
    print_partial_data({}, count_per_label=20, clinc_labels=["what_can_i_ask_you"])
    sys.exit()


def load_object_from_disk(complete_path: str):
    with open(complete_path, 'rb') as f:
        return pickle.load(f)


def load_pretrained_model_tokenizer(s3_params=None, model_name=None, save_path=None, model_directory=None,
                                    wikihow_data_json=None):
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
        id2label_save_path = "../saved_models/class_imbalance/id2label.pickle"
        label2id_save_path = "../saved_models/class_imbalance/label2id.pickle"
        if wikihow_data_json is not None:
            id2label_save_path = id2label_save_path.replace("id2label", "id2label18")
            label2id_save_path = label2id_save_path.replace("label2id", "label2id18")
        id2label = load_object_from_disk(id2label_save_path)
        label2id = load_object_from_disk(label2id_save_path)
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


local_model_directory = "../saved_models/multiclass_cfn2"


def load_pretrained_model(model_directory=local_model_directory, s3_param=None) \
        -> (PreTrainedModel, PreTrainedTokenizer):
    if s3_param:
        bucket = s3_param['bucket']
        path_to_model = s3_param['path_to_model']
        print('Loading model from S3. Bucket={}, path_to_model={}.'.format(bucket, path_to_model))
        model = load_model(bucket, path_to_model)
    else:
        print('Loading model from {}.'.format(model_directory))
        model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(model_directory)

    # Tokenizer is not fine-tuned so we get it from HuggingFace directly.
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    print('num labels = {}'.format(model.config.num_labels))
    return model, tokenizer