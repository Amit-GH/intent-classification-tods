
# Towards effective Task Oriented Dialogue Systems - Learning Intent Classification, Dialogue State Tracking and Question Answering
This repository supports the MS Thesis work and CS685 course project.

## FAQs
### 1. Where is the model training code?
All training and evaluation code is present in `classification/model.py`. Once models are trained and saved locally or
in S3, they are verified for their accuracy in `classification/model_check.py`.

### 2. Where is the dataset?
1. `data/data_full.json` has the entire data including train, val and test splits.
2. All intents along with domains are present in `data/domains.json`.
3. AP taskbot intent to clinc intent mapping is present in `data/intent_mapping.json`.
4. Examples user utterances for comparing different models are in `data/test_examples.json`.
5. Some low resource intents with sample utterances are present in `data/manual_data.json`.

### 3. Where is code to process dataset?
1. Data preprocessing code is present in `data_loader/DataLoader.py`.
2. Utility methods to interact with S3 is present in `data_loader/S3Loader.py`.

### 4. How to run the code?
1. Create the conda environment using the `environment.yml` file.
2. All python files (`model.py`, `model_check.py`, `DataLoader.py` and `S3Loader.py`) have `main` methods that were run
using IDE or command line to invoke specific functions. These functions are used for fine-tuning a model, saving a model
locally, uploading and downloading models from S3 bucket, calculating validation and test accuracy on a saved fine-tuned
model. There are methods for dataset creation (balanced or unbalanced) and training IC model with or without class 
weights.

All code present in `classification` and `data_loader` directories has been written specifically for this project.
