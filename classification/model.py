import json
import os
import sys

import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer, Trainer, PreTrainedModel, \
    PreTrainedTokenizer, DistilBertForSequenceClassification
from transformers import AutoModelForSequenceClassification, TrainingArguments
from data_loader.DataLoader import ClincDataSet, load_data, Group, load_mapped_data, ClincSingleData
from data_loader.S3Loader import load_model, upload_model
import torch
import wandb

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
local_model_directory = "../saved_models/multiclass_cfn"


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


def multi_class_classifier_accuracy(group: str, model_directory=local_model_directory, s3_params=None):
    model, tokenizer = load_pretrained_model(
        model_directory=model_directory,
        s3_param=s3_params if s3_params else None
    )
    configuration = model.config

    assert group in [Group.val.value, Group.test.value]

    mapped_data = load_mapped_data({})
    test_dataset = ClincDataSet(mapped_data[group][1], tokenizer)

    data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    acc_list = []
    label_to_stats = {}  # key=label, value=(correct_count, incorrect_count)

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

            labels_np = labels.numpy()
            preds_mask = 1 * (labels_np == preds)  # 1=correct, 0=incorrect
            for i, value in enumerate(preds_mask):
                right, wrong = label_to_stats.get(labels_np[i], (0, 0))
                if value == 1:
                    right += 1
                else:
                    wrong += 1
                label_to_stats[labels_np[i]] = (right, wrong)

            acc_list = np.append(acc_list, preds_mask)
            # print('running test acc = {:.3g}'.format(np.mean(acc_list)))

    print('Accuracy for {} = {:.3g}'.format(group, np.mean(acc_list)))
    print("label_name, id, accuracy, count")
    # sort by increasing accuracy.
    for label_id, stats in sorted(label_to_stats.items(), key=lambda x: x[1][0] / (x[1][0] + x[1][1])):
        print("{}, {}, {:.3g}, {}".format(
            configuration.id2label[label_id], label_id, stats[0] / (stats[0] + stats[1]), (stats[0] + stats[1])
        ))
    print('')


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
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            for k, v in batch.items():
                batch[k] = v.to(device)
            outputs = model(**batch)
            loss = criterion(outputs.logits, batch['labels'])
            val_running_loss += loss.item()
            _, preds = torch.max(outputs.logits, dim=1)
            val_running_correct += (preds == batch['labels']).sum().item()
            if num_batches == (i + 1):  # early stopping
                break
    val_loss = val_running_loss
    val_accuracy = 100. * val_running_correct / len(val_loader.dataset)
    if log_metric:
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}')
        wandb.log({
            'val_loss': val_loss,
            'val_acc': val_loss
        })
    return val_loss, val_accuracy


def train_classifier_with_unbalanced_data(save_locally=True, s3_params=None):
    batch_size = 512
    max_epochs = 1
    print_every = 2

    mapped_data = load_mapped_data({}, balance_split=False)

    id2label = mapped_data['id2label']
    label2id = mapped_data['label2id']

    model: DistilBertModel = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )
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
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(train_class_weights, dtype=torch.float32))

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
    wandb.init(project="ms-project-701", entity="amitgh", config=config)

    for epoch in tqdm(range(max_epochs), total=max_epochs):
        model.train()
        train_loss = 0
        for i, batch in enumerate(train_loader):
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
                print(f'train_batch_loss = {train_batch_loss}')
                wandb.log({'train_batch_loss': train_batch_loss})
                _, _ = perform_validation(model, val_loader, criterion, num_batches=2)
        print(f'Epoch = {epoch}, train loss = {train_loss}')
        wandb.log({'train_loss': train_loss})
        _, _ = perform_validation(model, val_loader, criterion, log_metric=True, num_batches=2)
    print('done')


def train_multi_class_classifier(save_locally=True, s3_params=None):
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


if __name__ == '__main__':
    # setup()
    # test_pretrained()
    # test_cuda()

    # train_multi_class_classifier(save_locally=False)

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
    # multi_class_classifier_accuracy(Group.val.value)

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

    train_classifier_with_unbalanced_data(save_locally=False)

    sys.exit()
