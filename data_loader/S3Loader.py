import os
import pickle
import tempfile

import boto3
import json

from contextlib import contextmanager
from io import BytesIO
from tempfile import NamedTemporaryFile

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, AutoModelForSequenceClassification

from data_loader.device_setup import device


@contextmanager
def s3_fileobj(bucket, key):
    """
    Yields a file object from the filename at {bucket}/{key}

    Args:
        bucket (str): Name of the S3 bucket where you model is stored
        key (str): Relative path from the base of your bucket, including the filename and extension of the object
            to be retrieved.
    """
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    yield BytesIO(obj["Body"].read())


def load_model_from_single_file(s3_params: dict, model_name: str):
    bucket = s3_params['bucket']
    path_to_model = s3_params['path_to_model']
    model = None

    with s3_fileobj(bucket, f'{path_to_model}/id2label.pickle') as f:
        id2label = pickle.loads(f.read())
    with s3_fileobj(bucket, f'{path_to_model}/label2id.pickle') as f:
        label2id = pickle.loads(f.read())

    empty_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )

    model_file = NamedTemporaryFile()
    with s3_fileobj(bucket, f'{path_to_model}/{model_name}') as f:
        model_file.write(f.read())

    model = load_model_from_disk(model_file.name, empty_model)
    return model


def load_model(bucket, path_to_model, model_name='pytorch_model') -> PreTrainedModel:
    """
    Load a model at the given S3 path. It is assumed that your model is stored at the key:

        '{path_to_model}/{model_name}.bin'

    and that a config has also been generated at the same path named:

        f'{path_to_model}/config.json'

    """
    model = None
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create temporary files for storing the bin (binary) and config (.json) files.
        # Their directory is different from tmpdirname.
        bin_file = NamedTemporaryFile()
        config_file = NamedTemporaryFile(mode="w+")

        # Download data from S3 and put it in the above created temporary files.
        with s3_fileobj(bucket, f'{path_to_model}/{model_name}.bin') as f:
            bin_file.write(f.read())

        with s3_fileobj(bucket, f'{path_to_model}/config.json') as f:
            json_obj = json.load(f)
            json.dump(json_obj, config_file)
            config_file.flush()

        # Temporary files have random names. We need the name to be specific so that huggingface loader
        # can understand it. We create symlink from new expected name to the original name.

        bin_file_exp_name = f'{tmpdirname}/{model_name}.bin'
        config_file_exp_name = f'{tmpdirname}/config.json'

        os.link(bin_file.name, bin_file_exp_name)
        os.link(config_file.name, config_file_exp_name)

        model = AutoModelForSequenceClassification.from_pretrained(tmpdirname)
    return model


def upload_pickle_files(s3_params: dict, local_file_path) -> bool:
    bucket = s3_params['bucket']
    path_to_model = s3_params['path_to_model']

    try:
        s3 = boto3.client("s3")
        file_name = local_file_path.split('/')[-1]
        s3.upload_file(local_file_path, bucket, f'{path_to_model}/{file_name}')
        return True
    except Exception as e:
        print(f'Got exception {e}')
        return False


def upload_model(s3_params: dict, local_directory_path=None, local_model_path=None):
    """
    Upload the model to S3.
    :param s3_params: dict containing bucket and path_to_model parameters.
    :param local_directory_path: Directory path that has config.json and pytorch_model.bin files.
    :param local_model_path: complete file path saving the model in a single file.
    :return: True if the model uploads successfully else False.
    """
    bucket = s3_params['bucket']
    path_to_model = s3_params['path_to_model']

    s3 = boto3.client("s3")
    try:
        if local_directory_path:
            s3.upload_file(f'{local_directory_path}/config.json', bucket, f'{path_to_model}/config.json')
            s3.upload_file(f'{local_directory_path}/pytorch_model.bin', bucket, f'{path_to_model}/pytorch_model.bin')
            print("S3 Upload Successful")
            return True
        elif local_model_path:
            model_name = local_model_path.split('/')[-1]
            s3.upload_file(local_model_path, bucket, f'{path_to_model}/{model_name}.pt')
            print("S3 Upload Successful")
            return True
    except Exception as e:
        print("Error in uploading model files. ", e)
        return False


def load_model_from_disk(save_path: str, empty_model: nn.Module) -> nn.Module:
    empty_model.load_state_dict(torch.load(save_path, map_location=device))
    empty_model.eval()
    print('Model loaded from path {} successfully.'.format(save_path))
    return empty_model


def test_temporary_directory():
    with tempfile.TemporaryDirectory() as tmpdirname:
        bin_file = NamedTemporaryFile(dir=tmpdirname, suffix='_pytorch_model.bin')
        config_file = NamedTemporaryFile(dir=tmpdirname, suffix='_config.json')
        print('')
    print('done')


if __name__ == '__main__':
    # s3://umass-alexaprize-model-hosting/multiclass_intent_cfn/
    # model = load_model('umass-alexaprize-model-hosting', 'multiclass_intent_cfn')
    # print('Model loaded from S3.')
    # test_temporary_directory()
    # upload_res = upload_model(
    #     local_directory_path="../saved_models/multiclass_cfn",
    #     s3_params={
    #         'bucket': "umass-alexaprize-model-hosting",
    #         'path_to_model': "multiclass_intent_cfn"
    #     }
    # )

    # Credentials needed if running in some random EC2 instance.
    os.environ.setdefault('AWS_ACCESS_KEY_ID', 'xxx')
    os.environ.setdefault('AWS_SECRET_ACCESS_KEY', 'yyy')

    # print('Uploading the model.')
    # upload_res = upload_model(
    #     s3_params={
    #         'bucket': "umass-alexaprize-model-hosting",
    #         'path_to_model': "test_folder"
    #     },
    #     local_model_path="../saved_models/class_imbalance/dict_values([0.0001, 15000, 10, 512, 5, 'DistilBertModel+Linear', 'modified-CLINC150'])"
    # )
    # print(f'Model uploaded: {upload_res}')

    # print('Uploading pickle files.')
    # upload_files = upload_pickle_files(
    #     s3_params={
    #         'bucket': "umass-alexaprize-model-hosting",
    #         'path_to_model': "test_folder"
    #     },
    #     local_file_path="../saved_models/class_imbalance/label2id.pickle"
    # )
    # print(f'Files uploaded: {upload_files}.')

    # TODO: error in below loading code.
    my_model = load_model_from_single_file(
        s3_params={
            'bucket': "umass-alexaprize-model-hosting",
            'path_to_model': "test_folder"
        },
        # model_name="dict_values([0.0001, 15000, 10, 512, 5, 'DistilBertModel+Linear', 'modified-CLINC150'].pt"
        model_name="weighted_class_model.pt"
    )
    print('done with setup')
