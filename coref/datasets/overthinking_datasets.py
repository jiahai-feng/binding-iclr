import datasets
import pandas as pd
import numpy as np

"""
From 
https://github.com/dannyallover/overthinking_the_truth/blob/31d40ad0533e7a143edb3db5efb1bb107bf8b3cd/data/dataset_params.py
https://github.com/dannyallover/overthinking_the_truth/blob/31d40ad0533e7a143edb3db5efb1bb107bf8b3cd/data/prompt_params.py

"""
AGNEWS_DATASET_PARAMS = {
    "set_name": "ag_news",
    "config": None,
    "train_or_test": "train",
    "on_hugging_face": True,
    "content_label_keys": (["text"], "label"),
}

AGNEWS_LABELS = [["World", "Sports", "Business", "Science"]]
AGNEWS_PROMPT_FORMATS = ["Article: {}\nAnswer: {}."]
AGNEWS_PREFIX_NARRATIVES = [""]
AGNEWS_PROMPT_PARAMS = {
    i: {
        "labels": AGNEWS_LABELS[i],
        "prompt_format": AGNEWS_PROMPT_FORMATS[i],
        "prefix_narrative": AGNEWS_PREFIX_NARRATIVES[i],
    }
    for i in range(len(AGNEWS_PROMPT_FORMATS))
}

SST2_DATASET_PARAMS = {
    "set_name": "sst2",
    "config": None,
    "train_or_test": "train",
    "on_hugging_face": True,
    "content_label_keys": (["sentence"], "label"),
}
SST2_LABELS = [
    ("Negative", "Positive"),
    ("bad", "good"),
    ("bad", "good"),
    ("negative", "positive"),
    ("bad", "good"),
    ("No", "Yes"),
    ("Negative", "Positive"),
    ("bad", "good"),
    ("bad", "good"),
    ("bad", "good"),
    ("liked", "hated"),
    ("0", "5"),
    ("Negative", "Positive"),
    ("False", "True"),
    ("0", "5"),
]
SST2_PROMPT_FORMATS = [
    "Review: {}\nAnswer: {}.",
    "Review: {}\nAnswer: {}.",
    "My review for last night’s film: {} The critics agreed that this movie was {}.",
    'One of our critics wrote "{}". Her sentiment towards the film was {}.',
    'In a contemporary review, Roger Ebert wrote: "{}". Entertainment Weekly  agreed, and the overall critical reception of the film was {}.',
    "Review: {}\nPositive Review? {}.",
    "Review: {}\nQuestion: Is the sentiment of the above review Positive or Negative?\nAnswer: {}.",
    "Review: {}\nQuestion: Did the author think that the movie was good or bad?\nAnswer: {}.",
    "Question: Did the author of the following tweet think that the movie was good or bad?\nTweet: {}\nAnswer: {}.",
    "{} My overall feeling was that the movie was {}.",
    "{} I {}.",
    "{} My friend asked me if I would give the movie 0 or 5 stars, I said {}.",
    "Input: {}\nSentiment: {}.",
    "Review: {}\nPositive: {}.",
    "Review: {}\nStars: {}.",
]
SST2_PREFIX_NARRATIVES = [
    "",
    "",
    "Here is what our critics think for this month’s films.",
    "Critical reception [ edit ].",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
]
SST2_PROMPT_PARAMS = {
    i: {
        "labels": SST2_LABELS[i],
        "prompt_format": SST2_PROMPT_FORMATS[i],
        "prefix_narrative": SST2_PREFIX_NARRATIVES[i],
    }
    for i in range(len(SST2_PROMPT_FORMATS))
}


def get_dataset(dataset_params: dict) -> list:
    """
    Get the dataset based off |dataset_params|.

    Parameters
    ----------
    dataset_params : required, dict
        Dataset metadata.

    Returns
    ------
    dataset : list
        List of examples for each class in the dataset.
    """
    set_name, config, train_or_test, on_hugging_face, content_label_keys = (
        dataset_params["set_name"],
        dataset_params["config"],
        dataset_params["train_or_test"],
        dataset_params["on_hugging_face"],
        dataset_params["content_label_keys"],
    )
    if on_hugging_face:
        raw_data_df = datasets.load_dataset(set_name, config)[train_or_test].to_pandas()
    else:
        raw_data_df = pd.read_csv(f"../../data/{set_name}.csv")

    content_keys, label_key = content_label_keys
    labels = raw_data_df[label_key].unique()
    labels.sort()

    dataset = []
    for l in labels:
        chained_content = [
            raw_data_df[raw_data_df[label_key] == l][content_keys[i]].values
            for i in range(len(content_keys))
        ]
        interleaved_content = list(zip(*chained_content))
        dataset.append(interleaved_content)

    return dataset
