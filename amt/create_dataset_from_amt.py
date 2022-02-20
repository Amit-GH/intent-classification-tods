import json
from collections import Counter

import pandas

from data_loader.DataLoader import load_amt_test_data

elaborate_label_to_int = {
    "Asks a recipe task": 1,
    "Asks a do-it-yourself task": 2,
    "Asks a task related or unrelated question": 3,
    "Asks a question about bot features": 4,
    "Asks an incomplete question": 5,
    "Shows acceptance with bot response": 6,
    "Does not agree with bot response": 7,
    "Shows acknowledgement": 8,
    "Asks a financial question or task": 9,
    "Asks for a legal advice": 10,
    "Asks for a medical advice": 11,
    "Tells something related to privacy": 12,
    "Tells something harmful": 13,
    "Tells something related to the shopping list": 14,
    "Asks to repeat the sentence": 15,
    "Asks to move ahead or tell the next step": 16,
    "Asks to go back or tell the previous statement": 17,
    "Asks to take a break or pause": 18,
    "Asks something related to ingredients": 19,
    "Tells to move to cooking steps": 20,
    "Suggests that the entire task has been completed": 21,
    "Suggests that one step has been completed": 22,
    "Wants to leave the bot or exit": 23,
    "Wants to know more available options": 24,
    "Selects one of the given options": 25,
    "Wants to have a general chit-chat": 26,
    "Says a greeting": 27,
    "Thanking or complimenting the bot": 28,
    "Undefined or none of the above intentions": 29
}

"""
undefined
accept
navigation_repeat
navigation_bye
recipe_question
intermediate_question
small_talk
financial
timer
thank_you
greeting
where_are_you_from
what_is_your_name
decline
ask_feature_question
who_made_you
shopping_list
"""

elaborate_label_to_cfn_label = {
    "Asks a recipe task": "recipe_question",
    "Asks a do-it-yourself task": "undefined",
    "Asks a task related or unrelated question": "intermediate_question",
    "Asks a question about bot features": "ask_feature_question",
    "Asks an incomplete question": "intermediate_question",
    "Shows acceptance with bot response": "accept",
    "Does not agree with bot response": "decline",
    "Shows acknowledgement": "accept",
    "Asks a financial question or task": "financial",
    "Asks for a legal advice": "undefined",
    "Asks for a medical advice": "undefined",
    "Tells something related to privacy": "undefined",
    "Tells something harmful": "undefined",
    "Tells something related to the shopping list": "shopping_list",
    "Asks to repeat the sentence": "undefined",
    "Asks to move ahead or tell the next step": "undefined",
    "Asks to go back or tell the previous statement": "undefined",
    "Asks to take a break or pause": "undefined",
    "Asks something related to ingredients": "recipe_question",
    "Tells to move to cooking steps": "undefined",
    "Suggests that the entire task has been completed": "undefined",
    "Suggests that one step has been completed": "undefined",
    "Wants to leave the bot or exit": "undefined",
    "Wants to know more available options": "undefined",
    "Selects one of the given options": "undefined",
    "Wants to have a general chit-chat": "small_talk",
    "Says a greeting": "greeting",
    "Thanking or complimenting the bot": "thank_you",
    "Undefined or none of the above intentions": "undefined"
}


def is_agree(arr, size) -> bool:
    if size == 2:
        return arr[0] == arr[1]
    elif size == 3:
        return arr[0] == arr[1] or arr[1] == arr[2] or arr[2] == arr[0]
    else:
        raise Exception("size more than 3 not supported.")


def create_dataset(csv_files: list[str], target_file=None):
    dataset = []
    for file_name in csv_files:
        dataset += create_dataset_from_one_file(file_name)

    # Find data statistics.
    label_count = dict()
    for single_data in dataset:
        label = single_data[1]
        label_count[label] = label_count.get(label, 0) + 1

    with open(target_file, "w") as outfile:
        json.dump({"test": dataset, "train": [], "val": []}, outfile)
    print(label_count)


def create_dataset_from_one_file(csv_file_path: str) -> list:
    labels_to_exclude = ["shopping_list"]
    assignments = 3
    result_df = pandas.read_csv(csv_file_path)

    temp = dict()
    accepted_hits = set()
    for hitid, label in zip(result_df["HITId"][:], result_df["Answer.intent.label"][:]):
        if hitid not in temp:
            temp[hitid] = []
        temp[hitid].append(elaborate_label_to_int[label])
        labels = temp[hitid]
        if len(labels) == assignments:
            if is_agree(labels, assignments):
                accepted_hits.add(hitid)

    added_hits = set()
    dataset = []
    label_count = dict()
    for hitid, label, text in zip(result_df["HITId"], result_df["Answer.intent.label"], result_df["Input.text"]):
        if hitid in accepted_hits:
            if hitid not in added_hits:
                added_hits.add(hitid)
                if elaborate_label_to_cfn_label[label] not in labels_to_exclude:
                    dataset.append([text, elaborate_label_to_cfn_label[label]])
                    label_count[elaborate_label_to_cfn_label[label]] = label_count.get(elaborate_label_to_cfn_label[label], 0) + 1

    print(label_count)
    return dataset


if __name__ == '__main__':
    create_dataset(["./Batch_4672852_batch_results.csv",  # Pilot3
                    "./Batch_4674004_batch_results.csv",  # Real1
                    "./Batch_4674090_batch_results.csv",  # Real2
                    "./Batch_4674199_batch_results.csv"],  # Real3
                   target_file="../data/amt_test_data_2.json")
