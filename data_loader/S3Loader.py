import os
import tempfile

import boto3
import json

from contextlib import contextmanager
from io import BytesIO
from tempfile import NamedTemporaryFile
from transformers import PretrainedConfig, PreTrainedModel, AutoModelForSequenceClassification


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


def upload_model(local_directory_path: str, s3_params: dict):
    bucket = s3_params['bucket']
    path_to_model = s3_params['path_to_model']

    s3 = boto3.client("s3")
    try:
        s3.upload_file(f'{local_directory_path}/config.json', bucket, f'{path_to_model}/config.json')
        s3.upload_file(f'{local_directory_path}/pytorch_model.bin', bucket, f'{path_to_model}/pytorch_model.bin')
        print("S3 Upload Successful")
        return True
    except Exception as e:
        print("Error in uploading model files. ", e)
        return False


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
    upload_res = upload_model("../saved_models/multiclass_cfn", {
        'bucket': "umass-alexaprize-model-hosting",
        'path_to_model': "test_file_upload"
    })
    print(upload_res)
