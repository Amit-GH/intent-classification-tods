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


def load_mapped_data(params: dict, balance_split=True):
    """
    To load clinc data into training, validation and testing which is mapped into custom classes as defined in
    the file intent_mapping.json.
    balance_split: bool: if true, we balance the number of examples of each class to be in the same range for all
        the three splits.
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
    # print('Total custom labels: {}\nCustom label names: {}'.format(len(label2id), label2id.keys()))

    # Load validation, train, test data.
    lc_val, val_data = _load_custom_data(clinc_data, clinc_to_custom, id2label, label2id, Group.val)
    lc_train, train_data = _load_custom_data(clinc_data, clinc_to_custom, id2label, label2id, Group.train)
    # print(sorted(lc_train.items(), key=lambda x: x[0]))
    lc_test, test_data = _load_custom_data(clinc_data, clinc_to_custom, id2label, label2id, Group.test)

    if balance_split:
        lc_val_bal, val_data_bal = balance_data(lc_val, val_data, 40, 100)
        lc_train_bal, train_data_bal = balance_data(lc_train, train_data, 200, 500)
        lc_test_bal, test_data_bal = balance_data(lc_test, test_data, 60, 150)
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


def load_model_from_disk(save_path: str, empty_model: nn.Module) -> nn.Module:
    empty_model.load_state_dict(torch.load(save_path))
    empty_model.eval()
    print('Model loaded from path {} successfully.'.format(save_path))
    return empty_model


def load_object_from_disk(complete_path: str):
    with open(complete_path, 'rb') as f:
        return pickle.load(f)