import numpy as np
from torch.utils.data import DataLoader
from transformers import DistilBertModel, DistilBertTokenizer, Trainer, PreTrainedModel, \
    PreTrainedTokenizer, DistilBertForSequenceClassification
from transformers import AutoModelForSequenceClassification, TrainingArguments
from data_loader.DataLoader import ClincDataSet, load_data, Group, load_mapped_data, ClincSingleData
from data_loader.S3Loader import load_model, upload_model
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
local_model_directory = "../saved_models/multiclass_cfn"


def load_pretrained_model(model_directory=local_model_directory, s3_param=None) \
        -> (PreTrainedModel, PreTrainedTokenizer):
    if s3_param:
        bucket = s3_param['bucket']
        path_to_model = s3_param['path_to_model']
        model = load_model(bucket, path_to_model)
    else:
        model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(model_directory)

    # Tokenizer is not fine-tuned so we get it from HuggingFace directly.
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    print('num labels = {}'.format(model.config.num_labels))
    return model, tokenizer


def analyze_custom_examples(sentences: list[str]):
    model, tokenizer = load_pretrained_model()
    configuration = model.config

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


def multi_class_classifier_accuracy(group: str, s3_params=None):
    model, tokenizer = load_pretrained_model(s3_param=s3_params if s3_params else None)
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
        num_train_epochs=1,  # total number of training epochs
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


def test_pretrained():
    model_directory = "../saved_models/binary_cfn"
    model: DistilBertModel = AutoModelForSequenceClassification.from_pretrained(model_directory)
    configuration = model.config
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    print('num labels = {}'.format(configuration.num_labels))

    sentences = [
        "For how long will this fish food last?",
        "how to make chicken korma?",
        "check my bank account.",
        "What is the time right now",
        "How to make tomato soup?"
    ]
    sentence_tokenize = tokenizer(sentences, return_tensors="pt", padding=True)

    res = model(**sentence_tokenize).logits  # (N, C)
    probs = torch.softmax(res, dim=1).tolist()  # (N, C)
    sorted_indices = np.flip(np.argsort(probs, axis=1), axis=1)  # (N, C)
    for i, indices in enumerate(sorted_indices):
        print(sentences[i])
        for idx in indices[:3]:
            print("{} {:.2g}%".format(configuration.id2label[idx], probs[i][idx] * 100))
        print()
    print('done')


def test_cuda():
    # Mac gpu cannot use cuda.
    print('cuda available:', torch.cuda.is_available())


if __name__ == '__main__':
    # setup()
    # test_pretrained()
    # test_cuda()

    # train_multi_class_classifier()
    # multi_class_classifier_accuracy(Group.test.value)
    multi_class_classifier_accuracy(
        Group.test.value,
        {
            'bucket': "umass-alexaprize-model-hosting",
            'path_to_model': "multiclass_intent_cfn"
        }
    )
    # multi_class_classifier_accuracy(Group.val.value)

    # analyze_predictions(Group.val.value, max_incorrect_count=20)

    sentences = [
        "For how long will this fish food last?",
        "how to make chicken korma?",
        "check my bank account.",
        "What is the time right now",
        "How to make tomato soup?"
    ]
    # analyze_custom_examples(sentences)
