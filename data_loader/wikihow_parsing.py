import csv
import json

import numpy as np
import pandas


def create_unique_titles(input_file_name: str, output_csv_file: str):
    df = pandas.read_csv(input_file_name)
    print(f"All column names = {df.columns.values.tolist()}")

    unique_titles = set()
    prefix_to_remove = "how to "

    for title in df["title"][:]:
        try:
            title = title.strip("0123456789").lower()
            if title.startswith(prefix_to_remove):
                title = title[len(prefix_to_remove):]
            unique_titles.add(title)
        except Exception as e:
            print(f"got exception {e} for title {title}.")

    unique_titles_list = list(unique_titles)
    rows = list(map(lambda x: [x], unique_titles_list))

    with open(output_csv_file, mode='w') as writefile:
        csv_writer = csv.writer(writefile)
        csv_writer.writerow(['title'])
        csv_writer.writerows(rows)


def convert_titles_to_json(input_csv_file: str, output_json_file: str, max_examples_to_use: int, class_name: str):
    df = pandas.read_csv(input_csv_file)
    titles = df["title"]
    shuffled_titles = np.random.permutation(titles.values)

    shuffled_titles = shuffled_titles[:max_examples_to_use]

    train_val_test_ratio = [10, 2, 3]  # the ratio present in Clinc150 dataset.
    train_size = int(len(shuffled_titles) * train_val_test_ratio[0] / sum(train_val_test_ratio))
    val_size = int(len(shuffled_titles) * train_val_test_ratio[1] / sum(train_val_test_ratio))
    test_size = len(shuffled_titles) - train_size - val_size

    train_data = shuffled_titles[:train_size]
    val_data = shuffled_titles[train_size:train_size+val_size]
    test_data = shuffled_titles[train_size+val_size:]

    wikihow_data = {
        "train": list(map(lambda x: [x, class_name], train_data)),
        "val": list(map(lambda x: [x, class_name], val_data)),
        "test": list(map(lambda x: [x, class_name], test_data))
    }

    with open(output_json_file, "w") as outfile:
        json.dump(wikihow_data, outfile)

    print("done")


if __name__ == '__main__':
    # create_unique_titles(input_file_name="../data/wikihowAll.csv",
    #                      output_csv_file="../data/wikihow_titles_lowercase.csv")

    convert_titles_to_json(input_csv_file="../data/wikihow_titles_lowercase.csv",
                           output_json_file="../data/wikihow_titles.json",
                           max_examples_to_use=110_250,
                           class_name="wikihow")

    print('done')