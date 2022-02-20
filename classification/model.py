import json
import os
import pickle
import sys
from enum import Enum

import numpy as np
import sklearn.metrics
from sklearn.metrics import precision_score, f1_score, recall_score
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer, Trainer
from transformers import AutoModelForSequenceClassification, TrainingArguments

from data_loader.DataLoader import ClincDataSet, load_data, Group, load_mapped_data, ClincSingleData, \
    load_pretrained_model_tokenizer, load_pretrained_model, \
    local_model_directory, load_amt_test_data
from data_loader.S3Loader import upload_model, load_model_from_disk
import torch
import wandb
from data_loader.device_setup import device

print(f'device={device}')


def persist_model(my_model, path_root="../saved_models/", append_name="") -> str:
    save_path = path_root + append_name
    torch.save(my_model.state_dict(), save_path)
    print("Model saved in path {}".format(save_path))
    return save_path


def analyze_custom_examples(sentences: list[str]):
    model, tokenizer = load_pretrained_model()
    configuration = model.config

    model.eval()
    with torch.no_grad():
        for sent in sentences:
            sentence_tokenize = tokenizer(sent, return_tensors="pt", padding=True)
            res = model(**sentence_tokenize).logits
            probs = torch.softmax(res, dim=1).tolist()[0]
            pred = np.argmax(probs)
            print("sent={}, pred={}".format(sent, configuration.id2label[pred]))


def analyze_predictions(group: str, max_incorrect_count=10, model_directory=None, s3_params=None):
    model, tokenizer = load_pretrained_model(s3_param=s3_params if s3_params else None)
    configuration = model.config

    assert group in [Group.val.value, Group.test.value]

    mapped_data = load_mapped_data({})
    test_data: list[ClincSingleData] = mapped_data[group][1]

    incorrect_count = 0

    model.eval()
    with torch.no_grad():
        for single_data in test_data:
            sentence_tokenize = tokenizer(single_data.sentence, return_tensors="pt", padding=True)
            res = model(**sentence_tokenize).logits
            probs = torch.softmax(res, dim=1).tolist()[0]
            pred = np.argmax(probs)
            if pred != single_data.label_id:
                print('exp={}, act={}, sent={}'.format(
                    configuration.id2label[single_data.label_id],
                    configuration.id2label[pred],
                    single_data.sentence
                ))
                incorrect_count += 1
                if incorrect_count >= max_incorrect_count:
                    break


def print_sklearn_metrics(y_true, y_pred):
    p_mi = precision_score(y_true, y_pred, average='micro', zero_division=0)
    p_ma = precision_score(y_true, y_pred, average='macro', zero_division=0)
    p_we = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"precision: micro={p_mi:.3g},"
          f"macro={p_ma:.3g}, "
          f"weighted={p_we:.3g}")

    r_mi = recall_score(y_true, y_pred, average='micro', zero_division=0)
    r_ma = recall_score(y_true, y_pred, average='macro', zero_division=0)
    r_we = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"recall: micro={r_mi:.3g},"
          f"macro={r_ma:.3g}, "
          f"weighted={r_we:.3g}")

    f1_mi = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_ma = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_we = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"f1-score: micro={f1_mi:.3g},"
          f"macro={f1_ma:.3g}, "
          f"weighted={f1_we:.3g}")

    print([p_mi, p_ma, p_we, r_mi, r_ma, r_we, f1_mi, f1_ma, f1_we])


def multi_class_classifier_accuracy(group: str, model_directory=local_model_directory, save_path=None,
                                    s3_params=None, balance_split=True, amt_input_file=None):
    model, tokenizer, _ = load_pretrained_model_tokenizer(s3_params, None, save_path, model_directory)
    model.to(device)
    configuration = model.config

    assert group in [Group.val.value, Group.test.value]

    if amt_input_file:
        test_data = load_amt_test_data(amt_input_file)
        test_dataset = ClincDataSet(test_data, tokenizer)
    else:
        mapped_data = load_mapped_data({}, balance_split=balance_split)
        test_dataset = ClincDataSet(mapped_data[group][1], tokenizer)

    data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    acc_list = []
    label_to_stats = {}  # key=label, value=(correct_count, incorrect_count)

    # We populate this to calculate sklearn metrics at the end.
    y_pred = np.array([])
    y_true = np.array([])

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            preds_softmax = torch.softmax(outputs.logits, dim=1).tolist()
            preds = np.argmax(preds_softmax, axis=1)

            labels_np = labels.cpu().numpy()
            preds_mask = 1 * (labels_np == preds)  # 1=correct, 0=incorrect
            y_true = np.append(y_true, labels_np)
            y_pred = np.append(y_pred, preds)
            for i, value in enumerate(preds_mask):
                right, wrong = label_to_stats.get(labels_np[i], (0, 0))
                if value == 1:
                    right += 1
                else:
                    wrong += 1
                label_to_stats[labels_np[i]] = (right, wrong)

            acc_list = np.append(acc_list, preds_mask)
            # print('running test acc = {:.3g}'.format(np.mean(acc_list)))

    print('N={}, Accuracy for {} = {:.3g}'.format(len(test_dataset), group, np.mean(acc_list)))
    balanced_accuracy = 0
    print("label_name, id, accuracy, count")
    # sort by increasing accuracy.
    for label_id, stats in sorted(label_to_stats.items(), key=lambda x: x[1][0] / (x[1][0] + x[1][1])):
        ba = stats[0] / (stats[0] + stats[1])
        print("{}, {}, {:.3g}, {}".format(
            configuration.id2label[label_id], label_id, ba, (stats[0] + stats[1])
        ))
        balanced_accuracy += ba
    balanced_accuracy /= len(label_to_stats)
    print("Balanced Accuracy = {:.3g}\n".format(balanced_accuracy))

    # l2s = dict()
    # for yt, yp in zip(y_true, y_pred):
    #     val = l2s.get(yt, [0, 0])
    #     l2s[yt] = [val[0] + 1 * (yt == yp), val[1] + 1 * (yt != yp)]

    print(f" *** sklearn metrics ***")
    print(f"array size = {len(y_pred)}")
    print_sklearn_metrics(y_true, y_pred)

    # Remove undefined class and calculate again.
    y_true_trim, y_pred_trim = [], []
    for yt, yp in zip(y_true, y_pred):
        if yt != 16:
            y_true_trim.append(yt)
            y_pred_trim.append(yp)

    print(f" *** sklearn metrics with undefined removed ***")
    print(f"array size = {len(y_pred_trim)}")
    print_sklearn_metrics(y_true_trim, y_pred_trim)


def calculate_class_weights(label_count: dict) -> list:
    """
    :param label_count: A dict where key is label_id and value is number of examples of that label.
    :return: A numpy list having weights for each class in order.
    """
    dict_items = list(label_count.items())
    dict_items.sort(key=lambda x: x[0])
    counts = np.array([count for (label_id, count) in dict_items])
    weighted_counts = counts.min() / counts
    return weighted_counts


def perform_validation(model, val_loader, criterion, log_metric=False, num_batches=None):
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    label_total_count = np.zeros(len(model.config.id2label))
    label_correct_count = np.zeros_like(label_total_count)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            for k, v in batch.items():
                batch[k] = v.to(device)
            outputs = model(**batch)
            loss = criterion(outputs.logits, batch['labels'])
            val_running_loss += loss.item()
            _, preds = torch.max(outputs.logits, dim=1)
            val_running_correct += (preds == batch['labels']).sum().item()

            # Calculate label wise accuracy data.
            for label_id in batch['labels']:
                label_total_count[label_id] += 1
            for j, is_correct in enumerate(preds == batch['labels']):
                if is_correct:
                    label_correct_count[batch['labels'][j]] += 1

            if num_batches == (i + 1):  # early stopping
                break
    val_loss = val_running_loss
    val_accuracy = 100. * val_running_correct / len(val_loader.dataset)
    if log_metric:
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}')
        label_accuracy = label_correct_count / label_total_count * 100
        for i, acc in enumerate(label_accuracy):
            print(f'id: {i}, label: {model.config.id2label[i]}, acc: {acc:.2f}')
        wandb.log({
            'val_loss': val_loss,
            'val_acc': val_accuracy
        })

    return val_loss, val_accuracy


def persist_object_to_disk(obj, complete_path: str):
    with open(complete_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def train_classifier_with_unbalanced_data(
        max_epochs=1, batch_size=512, print_every=2,
        save_locally=True, s3_params=None,
        start_from_pretrained=False, save_path=None,
        wandb_mode="online",
        balance_split=False, balance_strategy=None
):
    mapped_data = load_mapped_data({}, balance_split=balance_split, balance_strategy=balance_strategy)

    id2label = mapped_data['id2label']
    label2id = mapped_data['label2id']

    model: DistilBertModel = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )
    # Save model metadata to disk for later usage.
    persist_object_to_disk(id2label, "../saved_models/class_imbalance/id2label.pickle")
    persist_object_to_disk(label2id, "../saved_models/class_imbalance/label2id.pickle")

    if start_from_pretrained:
        model, _, _ = load_pretrained_model_tokenizer(save_path=save_path)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    train_dataset = ClincDataSet(mapped_data[Group.train.value][1], tokenizer)
    val_dataset = ClincDataSet(mapped_data[Group.val.value][1], tokenizer)
    test_dataset = ClincDataSet(mapped_data[Group.test.value][1], tokenizer)

    train_class_weights = calculate_class_weights(mapped_data[Group.train.value][0])
    val_class_weights = calculate_class_weights(mapped_data[Group.val.value][0])
    test_class_weights = calculate_class_weights(mapped_data[Group.test.value][0])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    model.to(device)

    # Create a weighted loss metric. Keep type similar to that of the model.
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(train_class_weights, dtype=torch.float32, device=device))

    optimizer = optim.AdamW(model.parameters(),
                            lr=1e-4,
                            eps=1e-5
                            )

    # Initialize wandb and add variables that you want associate with this run.
    os.environ.setdefault('WANDB_API_KEY', '713a778aae8db6219a582a6b794204a5af2cb75d')
    config = {
        "learning_rate": 1e-4,
        "train_size": len(train_loader.dataset),
        "epochs": max_epochs,
        "batch_size": batch_size,
        "print_every": print_every,
        "architecture": "DistilBertModel+Linear",
        "dataset": "modified-CLINC150"
    }
    wandb.init(project="ms-project-701", entity="amitgh", config=config, mode=wandb_mode)

    best_val_acc = 0

    for epoch in tqdm(range(max_epochs), total=max_epochs):
        model.train()
        train_loss = 0
        for i, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            for k, v in batch.items():
                batch[k] = v.to(device)

            outputs = model(**batch)
            loss = criterion(outputs.logits, batch['labels'])
            loss.backward()
            optimizer.step()

            train_batch_loss = loss.item()
            train_loss += train_batch_loss
            if (i + 1) % print_every == 0:
                _, preds = torch.max(outputs.logits, dim=1)
                train_batch_acc = (preds == batch['labels']).to(torch.float32).mean().item() * 100
                print(f'train_batch_loss = {train_batch_loss}, train_batch_acc = {train_batch_acc}')
                wandb.log({
                    'train_batch_loss': train_batch_loss,
                    'train_batch_acc': train_batch_acc
                })
                # _, _ = perform_validation(model, val_loader, criterion, log_metric=True)
        print(f'Epoch = {epoch}, train loss = {train_loss}')
        wandb.log({'train_loss': train_loss})
        val_loss, val_acc = perform_validation(model, val_loader, criterion, log_metric=True)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_locally:
                persist_model(model, append_name=f"class_imbalance/split-{balance_split}-strategy-{balance_strategy}-{config.values()}.pt")
    wandb.finish()
    print('done')


def train_multi_class_classifier(save_locally=True, s3_params=None):
    os.environ.setdefault('WANDB_API_KEY', '713a778aae8db6219a582a6b794204a5af2cb75d')
    wandb.login()

    mapped_data = load_mapped_data({})

    id2label = mapped_data['id2label']
    label2id = mapped_data['label2id']

    model: DistilBertModel = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )
    configuration = model.config
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    train_dataset = ClincDataSet(mapped_data[Group.train.value][1], tokenizer)
    val_dataset = ClincDataSet(mapped_data[Group.val.value][1], tokenizer)
    test_dataset = ClincDataSet(mapped_data[Group.test.value][1], tokenizer)

    training_args = TrainingArguments(
        output_dir='../output_dir',  # output directory
        num_train_epochs=5,  # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=100,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='../output_dir/logs',  # directory for storing logs
        logging_steps=10,
        report_to="wandb",
        run_name="balanced_multiclass_cfn_train"
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset  # evaluation dataset
    )

    trainer.train()

    if save_locally:
        model.save_pretrained(local_model_directory)
        print('Model saved locally.')
        if s3_params:
            upload_res = upload_model(
                local_directory_path=local_model_directory,
                s3_params=s3_params
            )
            if upload_res:
                print('Model saved remotely on S3.')


def setup():
    combined_data = load_data({})
    id_to_label = combined_data['id_to_label']
    label_to_id = combined_data['label_to_id']

    model: DistilBertModel = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(id_to_label),
        id2label=id_to_label,
        label2id=label_to_id
    )
    configuration = model.config
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # TODO: freezing bert model https://github.com/huggingface/transformers/issues/400
    # TODO: Considering adding one more layer in the FC classification head.

    _, train_data_raw = combined_data[Group.train.value]
    train_dataset = ClincDataSet(train_data_raw[:], tokenizer=tokenizer)
    _, val_data_raw = combined_data[Group.val.value]
    val_dataset = ClincDataSet(val_data_raw[:], tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=5,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset  # evaluation dataset
    )

    trainer.train()

    model.save_pretrained("../saved_models/binary_cfn")


def fine_tune_model(params: dict, s3_params=None, save_locally=False, model_directory=None):
    """
    Download model from S3.
    Fine-tune it on some data. Try partial model freezing.
    Check its val and test accuracies.
    Save it locally.
    """

    model, tokenizer = load_pretrained_model(s3_param=s3_params if s3_params else None)
    configuration = model.config

    mapped_data = load_mapped_data({})

    id2label = mapped_data['id2label']
    label2id = mapped_data['label2id']

    val_dataset = ClincDataSet(mapped_data[Group.val.value][1], tokenizer)
    test_dataset = ClincDataSet(mapped_data[Group.test.value][1], tokenizer)

    train_data = mapped_data[Group.train.value][1]

    # Load the fine-tuning data.
    root_path = params.get('root_path', '../../')
    fine_tuning_data_path = os.path.join(root_path, 'intent-classification-tods/data/fine_tuning_data.json')
    fine_tuning_data: dict = json.load(open(fine_tuning_data_path, "r"))
    fine_tuning_clinc_data = []
    for label, example_list in fine_tuning_data.items():
        for example in example_list:
            fine_tuning_clinc_data.append(ClincSingleData(example, label2id[label], Group.train))
    # Add some data from training data too to balance things out.
    fine_tuning_clinc_data += train_data[:20]

    train_dataset = ClincDataSet(fine_tuning_clinc_data, tokenizer)

    training_args = TrainingArguments(
        output_dir='../output_dir',  # output directory
        num_train_epochs=20,  # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=100,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='../output_dir/logs',  # directory for storing logs
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset  # evaluation dataset
    )

    model.train()
    for internal_params in model.distilbert.parameters():
        internal_params.requires_grad = False

    trainer.train()

    if save_locally and model_directory:
        model.save_pretrained(model_directory)
        print('Fine-tuned model saved locally at {}.'.format(model_directory))


def test_cuda():
    # Mac gpu cannot use cuda.
    print('cuda available:', torch.cuda.is_available())


class WandbMode(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    DISABLED = "disabled"


if __name__ == '__main__':
    # setup()
    # test_pretrained()
    # test_cuda()

    # os.environ.set("WANDB_DISABLED", "true")

    # train_multi_class_classifier(save_locally=True)

    # Current S3 model: Accuracy for val = 0.95, Accuracy for test = 0.955
    # Complete fine-tuned model ep=1: Accuracy for val = 0.955, Accuracy for test = 0.957
    # Partial fine-tuned model ep=10: Accuracy for val = 0.955, Accuracy for test = 0.96
    # Partial fine-tuned model ep=20: Accuracy for val = 0.955, Accuracy for test = 0.952
    # multi_class_classifier_accuracy(Group.test.value, model_directory="../saved_models/fine_tuned_cfn")
    # multi_class_classifier_accuracy(Group.val.value, model_directory="../saved_models/fine_tuned_cfn")
    # multi_class_classifier_accuracy(
    #     group=Group.val.value,
    #     s3_params={
    #         'bucket': "umass-alexaprize-model-hosting",
    #         'path_to_model': "multiclass_intent_cfn"
    #     }
    # )
    # multi_class_classifier_accuracy(Group.test.value, balance_split=False)

    # analyze_predictions(Group.val.value, max_incorrect_count=20)

    # sentences = [
    #     "For how long will this fish food last?",
    #     "how to make chicken korma?",
    #     "check my bank account.",
    #     "What is the time right now",
    #     "How to make tomato soup?"
    # ]
    # analyze_custom_examples(sentences)

    # fine_tune_model(
    #     params={},
    #     s3_params={'bucket': "umass-alexaprize-model-hosting", 'path_to_model': "multiclass_intent_cfn"},
    #     save_locally=True,
    #     model_directory="../saved_models/fine_tuned_cfn"
    # )

    # train_classifier_with_unbalanced_data(max_epochs=10, print_every=5, save_locally=True,
    #                                       wandb_mode=WandbMode.ONLINE.value,
    #                                       balance_split=False, balance_strategy=None)

    save_path_list = [
        ("D2 with downsampling", "../saved_models/class_imbalance/split-True-strategy-down-dict_values-1700-20-512-5.pt"),
        ("D2 with upsampling", "../saved_models/class_imbalance/split-True-strategy-up-dict_values([0.0001, 156400, 1, 512, 5, 'DistilBertModel+Linear', 'modified-CLINC150']).pt"),
        ("D2 with balanced classes", "../saved_models/class_imbalance/split-True-strategy-None-dict_values([0.0001, 4600, 10, 512, 5, 'DistilBertModel+Linear', 'modified-CLINC150']).pt"),
        ("D2 with weighted loss", "../saved_models/class_imbalance/split-False-strategy-None-dict_values([0.0001, 15000, 10, 512, 5, 'DistilBertModel+Linear', 'modified-CLINC150']).pt")
    ]

    # Do evaluation on AMT test data.
    # for description, save_path in save_path_list:
    #     print(f"  *** {description} ***")
    #     multi_class_classifier_accuracy(
    #         Group.test.value, balance_split=False,
    #         save_path=save_path,
    #         amt_input_file="../data/amt_test_data_2.json"
    #     )
    #     print("\t\t*****\n\n\n")

    # Do evaluation on Dataset test data.
    for description, save_path in save_path_list:
        print(f"  *** {description} ***")
        multi_class_classifier_accuracy(
            Group.test.value, balance_split=False,
            save_path=save_path,
        )
        print("\t\t*****\n\n\n")
    sys.exit()
