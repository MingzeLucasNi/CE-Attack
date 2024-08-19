from transformers import pipeline
from datasets import load_dataset

def load_model(dataset, model_type):
    """
    Load the appropriate model based on the specified dataset and model type.
    
    Args:
        dataset (str): The name of the dataset, one of ["ag_news", "imdb", "sst2"] for classifiers or "wmt1", "wmt2" for NMT tasks.
        model_type (str): The type of model to load, either "classifier" or "nmt".
    
    Returns:
        A Hugging Face Pipeline object ready for text classification or translation tasks.
    """
    if model_type == "classifier":
        if dataset == "ag_news":
            model_name = "textattack/bert-base-uncased-ag-news"
        elif dataset == "imdb":
            model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        elif dataset == "sst2":
            model_name = "textattack/roberta-base-SST-2"
        else:
            raise ValueError("Unsupported classifier dataset specified.")
        
        # Return a text classification pipeline
        model = pipeline('sentiment-analysis', model=model_name)
    
    elif model_type == "nmt":
        if dataset == "wmt1":
            model_name = "Helsinki-NLP/opus-mt-en-fr"  # Example: English to French
        elif dataset == "wmt2":
            model_name = "Helsinki-NLP/opus-mt-en-de"  # Example: English to German
        else:
            raise ValueError("Unsupported NMT dataset specified.")
        
        # Return a translation pipeline
        model = pipeline('translation', model=model_name)
    
    else:
        raise ValueError("Unsupported model type specified.")
    
    return model

def load_dataset(dataset_name):
    """
    Load a dataset from Hugging Face datasets library.
    
    Args:
        dataset_name (str): The name of the dataset to load, such as "ag_news", "imdb", "sst2", "wmt1", or "wmt2".
    
    Returns:
        A dataset object from the Hugging Face datasets library.
    """
    if dataset_name not in ["ag_news", "imdb", "sst2", "wmt1", "wmt2"]:
        raise ValueError("Unsupported dataset specified.")
    
    if dataset_name in ["wmt1", "wmt2"]:
        dataset = load_dataset("wmt14", dataset_name)
    else:
        dataset = load_dataset(dataset_name)
    
    return dataset

def get_sample(dataset, split="test", index=0):
    """
    Retrieve a single sample from the specified dataset split.
    
    Args:
        dataset: The dataset object loaded from Hugging Face.
        split (str): The split to retrieve the sample from, default is "test".
        index (int): The index of the sample to retrieve.
    
    Returns:
        A tuple containing the text and its corresponding label or translation reference.
    """
    sample = dataset[split][index]
    
    if "text" in sample:
        return sample["text"], sample["label"]
    elif "sentence" in sample:
        return sample["sentence"], sample["label"]
    elif "content" in sample:
        return sample["content"], sample["label"]
    elif "translation" in sample and isinstance(sample["translation"], dict):
        return sample["translation"]["en"], sample["translation"]["fr"]  # Assuming English to French translation
    else:
        raise ValueError("Unsupported data format in the dataset.")

def prepare_data_and_model(dataset_name, model_type):
    """
    Prepare the dataset and model for use in the attack.
    
    Args:
        dataset_name (str): The name of the dataset to load, such as "ag_news", "imdb", "sst2", "wmt1", or "wmt2".
        model_type (str): The type of model to load, either "classifier" or "nmt".
    
    Returns:
        Tuple: A dataset object and a Hugging Face Pipeline model.
    """
    dataset = load_dataset(dataset_name)
    model = load_model(dataset_name, model_type)
    return dataset, model