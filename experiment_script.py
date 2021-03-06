#!/usr/bin/env python3

"""
usage: nlu-experiment [-h] [-mo MODEL_OUTPUT] [-t] [-e] [-mi MODEL_INPUT] [-nc] [-b BATCH_SIZE]
                      [-ed {EMBO/biolang,empathetic_dialogues,conv_ai_3,air_dialogue,ted_talks_iwslt,tweet_eval}] [--eval-all]

Used to train models and evaluate datasets

optional arguments:
  -h, --help            show this help message and exit
  -mo MODEL_OUTPUT, --model_output MODEL_OUTPUT
                        Where to store a trained model
  -t, --train           Train a model
  -e, --evaluate        Run an evaluation (either requires --train or a model_input)
  -mi MODEL_INPUT, --model-input MODEL_INPUT
                        Where to load a model (for evaluation purposes)
  -nc, --no-cuda        Whether to use cuda or not. Defaults to true. Turn off for debugging purposes
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size to use for training
  -ed {EMBO/biolang,empathetic_dialogues,conv_ai_3,air_dialogue,ted_talks_iwslt,tweet_eval}, --eval-dataset {EMBO/biolang,empathetic_dialogues,conv_ai_3,air_dialogue,ted_talks_iwslt,tweet_eval}
                        Dataset to evaluate on
  --eval-all            Whether to run evaluation on all datasets. Overrides all other eval commands
  -tc, --training-config
                        A path to a file containing a JSON blob containing config settings for model training
  -cd, --cache_dir      A path to a directory to use as a cache directory for loading datasets from Hugging Face

"""

metric_name = "f1"
batch_size = 2


from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    # EarlyStoppingCallback,
)

from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
import numpy as np
import json
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from pprint import pformat
import gc
import random
import argparse
import logging
from datetime import datetime
from relabel_funcs import (
    relabel_sbic_offensiveness,
    split_relabel_rt_gender,
    filter_relabel_sbic_targetcategory,
    split_relabel_jigsaw_toxic,
    split_relabel_jigsaw_severetoxic,
    split_relabel_jigsaw_identityhate,
    split_relabel_eec,
    relabel_md_gender_convai_binary,
    relabel_md_gender_convai_ternary,
    relabel_md_gender_wizard,
    relabel_md_gender_yelp,
)
import time
import spacy
from text_preprocess import keep_sentence, normalize_raw
from tqdm import tqdm
import os


nlp = spacy.load("en_core_web_sm")
logging.basicConfig(level=logging.INFO)
USE_CUDA = False
PICKLE_PATH = os.getcwd() + "/pickle_data/*"
# mapping of training datasets to their labels
training_dataset_cols = {
    "peixian/rtGender": "",
    "mdGender": "",
    "jigsaw_toxicity_pred": "toxic",
    "social_bias_frames": "offensiveYN",
    "peixian/equity_evaluation_corpus": "",
}


# mapping of training datasets to functions to relabel
training_relabel_funcs = {
    "relabel_sbic_offensiveness": relabel_sbic_offensiveness,
    "filter_relabel_sbic_targetcategory": filter_relabel_sbic_targetcategory,
    "split_relabel_rt_gender": split_relabel_rt_gender,
    "mdGender": "",
    "split_relabel_jigsaw_toxic": split_relabel_jigsaw_toxic,
    "split_relabel_jigsaw_severetoxic": split_relabel_jigsaw_severetoxic,
    "split_relabel_jigsaw_identityhate": split_relabel_jigsaw_identityhate,
    "split_relabel_eec": split_relabel_eec,
    "relabel_md_gender_convai_binary": relabel_md_gender_convai_binary,
    "relabel_md_gender_convai_ternary": relabel_md_gender_convai_ternary,
    "relabel_md_gender_wizard": relabel_md_gender_wizard,
    "relabel_md_gender_yelp": relabel_md_gender_yelp,
}


## mapping of dataset_name to dataset columns of sentences
dataset_cols = {
    "EMBO/biolang": "input_text",
    "empathetic_dialogues": "utterance",
    "conv_ai_3": "answer",
    "air_dialogue": "dialogue",
    "ted_talks_iwslt": "translation",
    "tweet_eval": "text",
    "cnn_dailymail": "highlights",
    "pubmed": "",
    "wikipedia": "title",
    "yelp_review_full": "text",
    "yahoo_answers_topics": "best_answer",
    "pubmed_qa": "long_answer",
    "billsum": "summary",
}


dataset_types = {
    "ted_talks_iwslt": ["nl_en_2014", "nl_en_2015"],
    "cnn_dailymail": ["1.0.0", "2.0.0", "3.0.0"],
    "wikipedia": ["20200501.en"],
    "tweet_eval": [
        "emoji",
        "emotion",
        "hate",
        "irony",
        "offensive",
        "sentiment",
        "stance_abortion",
        "stance_atheism",
        "stance_climate",
        "stance_feminist",
        "stance_hillary",
    ],
    "pubmed_qa": ["pqa_unlabeled"],
}


dataset_preprocess = {
    # 'NAME: <SENTENCE>'
    "ted_talks_iwslt": lambda x: x["en"],
}


split_para = set(
    [
        "ted_talks_iwslt",
        "empathetic_dialogues",
        "wikipedia",
        "yelp_review_full",
        "yahoo_answers_topics",
        "pubmed_qa",
        "billsum",
    ]
)


def get_sentences(paragraph):
    result = []
    try:
        doc = nlp(paragraph)
        for sentence in doc.sents:
            result.append(str(sentence))
    except Exception:
        print("This paragraph could not be converted", paragraph)
    return result


def split_long_text(list_paragraphs):
    results = []
    for paragraph in tqdm(list_paragraphs):
        results.append(get_sentences(paragraph))

    results = [item for sublist in results for item in sublist]
    return results


def concate(dataset_name, data, cache_dir):
    if dataset_name in dataset_types:
        all_datasets_downloaded = [
            load_dataset(dataset_name, sub_dataset, cache_dir=cache_dir)
            for sub_dataset in dataset_types[dataset_name]
        ]
        combined_datasets = [
            concatenate_datasets(list(sub_dataset.values()))
            for sub_dataset in all_datasets_downloaded
        ]
        data = concatenate_datasets(combined_datasets)
        return DatasetDict({"train": data})
    data = concatenate_datasets(
        list(load_dataset(dataset_name, cache_dir=cache_dir).values())
    )
    return DatasetDict({"train": data})


def loader(dataset_name, tokenizer, cache_dir, short_test):
    assert dataset_name in dataset_cols
    sentence_col = dataset_cols[dataset_name]
    d_types = dataset_types.get(dataset_name, None)
    tot = []
    logging.info(f"Using cache dir {cache_dir}")
    if d_types:
        for d_type in d_types:
            data = load_dataset(dataset_name, d_type, cache_dir=cache_dir)
            tot.append(
                _preprocess_dataset(
                    dataset_name,
                    data,
                    sentence_col,
                    tokenizer,
                    cache_dir=cache_dir,
                    short_test=short_test,
                )
            )
    else:
        data = load_dataset(dataset_name, cache_dir=cache_dir)
        tot.append(
            _preprocess_dataset(
                dataset_name,
                data,
                sentence_col,
                tokenizer,
                cache_dir=cache_dir,
                short_test=short_test,
            )
        )
    return tot


def _preprocess_dataset(
    dataset_name, data, sentence_col, tokenizer, cache_dir="", short_test=False
):
    preprocess_function = dataset_preprocess.get(dataset_name, lambda x: x)
    data = concate(dataset_name, data, cache_dir)

    data = data.map(lambda x: {"input_text": preprocess_function(x[sentence_col])})
    data["train"] = data["train"].remove_columns(
        set(data["train"].features) - set(["input_text"])
    )

    logging.info(f"NP Concate")
    if dataset_name == "air_dialogue":
        data["train"] = Dataset.from_dict(
            {"input_text": np.concatenate(data["train"]["input_text"]).ravel().tolist()}
        )

    if short_test:
        data["train"] = Dataset.from_dict(
            {"input_text": data["train"]["input_text"][:30]}
        )

    if dataset_name == "air_dialogue" or dataset_name == "yahoo_answers_topics":
        data["train"] = Dataset.from_dict(
            {"input_text": data["train"]["input_text"][:100000]}
        )
    elif dataset_name == "wikipedia" or dataset_name == "yelp_review_full":
        data["train"] = Dataset.from_dict(
            {"input_text": data["train"]["input_text"][:200000]}
        )

    if dataset_name in split_para:
        logging.info(f"Splitting Paragraphs")
        data["train"] = Dataset.from_dict(
            {"input_text": split_long_text(data["train"]["input_text"])}
        )

    logging.info(f"Normalize")
    data = data.map(lambda x: {"input_text": normalize_raw(x["input_text"])})
    logging.info(f"Keep Sentence")
    data = data.filter(lambda x: keep_sentence(x["input_text"]))

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    filename = f"{dataset_name}-full-text.out"
    logging.info(f"Opening file {filename} to write results")
    with open(filename, "w") as outfile:
        outfile.write("FULL TEXT BELOW\n")
        for i, text in enumerate(data["train"]["input_text"]):
            outfile.write(f"{i} | {text}\n")

    logging.info(f"Join")
    data = data.map(lambda x: {"input_text": " ".join(x["input_text"])})
    logging.info(f"Tokenizer")
    data = data.map(
        lambda x: tokenizer(x["input_text"], padding="max_length", truncation=True),
        batched=True,
    )
    return data


def compute_metrics(pred, average_fscore_support):
    preds, labels = pred
    preds = np.argmax(preds, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=average_fscore_support
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def train(model, encoded_dataset, tokenizer, run_output_dir, average_fscore_support):
    args = TrainingArguments(
        output_dir=run_output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        learning_rate=1.218154152691866e-05,
        num_train_epochs=2,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        greater_is_better=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=lambda pred: compute_metrics(pred, average_fscore_support),
    )
    logging.info("Constructed trainer")

    trainer.train()

    return trainer


def make_chunks(data1, data2, data3, chunk_size):
    while data1 and data2:
        chunk1, data1 = data1[:chunk_size], data1[chunk_size:]
        chunk2, data2 = data2[:chunk_size], data2[chunk_size:]
        chunk3, data3 = data3[:chunk_size], data3[chunk_size:]
        yield chunk1, chunk2, chunk3


def collate_fn(examples):
    return tokenizer.pad(
        examples, padding="max_length", truncation=True, return_tensors="pt"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="nlu-experiment", description="Used to train models and evaluate datasets"
    )
    parser.add_argument(
        "-mo", "--model_output", default="./", help="Where to store a trained model"
    )
    parser.add_argument(
        "-t", "--train", action="store_true", default=False, help="Train a model"
    )
    parser.add_argument(
        "-td",
        "--train-dataset",
        default=None,
        choices=training_dataset_cols.keys(),
        help="Which dataset to train on",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        action="store_true",
        default=False,
        help="Run an evaluation (either requires --train or a model_input)",
    )
    parser.add_argument(
        "-mi",
        "--model-input",
        default="",
        help="Where to load a model (for evaluation purposes)",
    )
    parser.add_argument(
        "-nc",
        "--no-cuda",
        action="store_true",
        default=False,
        help="Whether to use cuda or not. Defaults to true. Turn off for debugging purposes",
    )
    parser.add_argument(
        "-b", "--batch-size", default=2, help="Batch size to use for training"
    )

    parser.add_argument(
        "-cd",
        "--cache-dir",
        default=None,
        required=True,
        help="Cache dir to use for datasets to download into. Usually best to set to the value of $SCRATCH/datasets",
    )

    parser.add_argument(
        "-ed",
        "--eval-dataset",
        default=None,
        choices=dataset_cols.keys(),
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--eval-all",
        action="store_true",
        default=False,
        help="Whether to run evaluation on all datasets. Overrides all other eval commands",
    )

    parser.add_argument(
        "-tc",
        "--training-config",
        default=None,
        required=True,
        help="A path to a file containing a JSON blob containing config settings for model training",
    )
    parser.add_argument(
        "--short-test",
        default=False,
        required=False,
        action="store_true",
        help="only run first 10 sentences within eval dataset",
    )

    args = parser.parse_args()
    if not args.train and not args.evaluate:
        print("Not training AND not evaluating. Exiting.")
        exit(0)
    if args.evaluate and not args.train and not args.model_input:
        print(
            "Asked for evaluation but we are not training a model or loading one. Please supply a model or train one."
        )

    USE_CUDA = not args.no_cuda

    logging.getLogger().setLevel(logging.INFO)

    logging.info("Initializing seeds and setting values")
    logging.info(f"Use cuda? {USE_CUDA}")
    gc.collect()
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    if USE_CUDA:
        torch.cuda.empty_cache()

    logging.info(
        f"Loading dictionary of training parameters from {args.training_config}"
    )

    with open(args.training_config, "r") as f:

        training_config_dict = json.load(f)

    AVERAGE_FSCORE_SUPPORT = training_config_dict["average_fscore_support"]
    TRAINING_DATASET = training_config_dict["training_dataset"]
    TRAINING_DATASET_SPLIT = (
        None
        if training_config_dict["training_dataset_split"] == "None"
        else training_config_dict["training_dataset_split"]
    )
    MODEL_CHECKPOINT = training_config_dict["model_checkpoint"]
    RUN_OUTPUTS = training_config_dict["run_outputs"]
    TRAIN_FEATURES_COLUMN = training_config_dict["train_features_column"]
    NUM_LABELS = training_config_dict["num_labels"]
    TRAIN_LABELS_COLUMN = training_config_dict["train_labels_column"]
    TRAINING_RELABEL_FUNC_NAME = training_config_dict["training_relabel_func_name"]
    DATA_DIR = None
    SUBCORPUS = None

    if TRAINING_DATASET == "jigsaw_toxicity_pred":

        DATA_DIR = training_config_dict["jigsaw_dataset_dir"]

    if TRAINING_DATASET in set(["peixian/rtGender", "md_gender_bias"]):

        SUBCORPUS = training_config_dict["subcorpus"]

    logging.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)
    if args.train:
        CACHE_DIR = args.cache_dir
        relabel_training = training_relabel_funcs[TRAINING_RELABEL_FUNC_NAME]
        if args.train_dataset:
            TRAINING_DATASET = args.train_dataset
        logging.info(
            f"Loading dataset {TRAINING_DATASET} with split {TRAINING_DATASET_SPLIT}"
        )
        dataset = (
            load_dataset(
                TRAINING_DATASET,
                split=TRAINING_DATASET_SPLIT,
                cache_dir=CACHE_DIR,
                data_dir=DATA_DIR,
            )
            if SUBCORPUS is None
            else load_dataset(
                TRAINING_DATASET,
                SUBCORPUS,
                split=TRAINING_DATASET_SPLIT,
                cache_dir=CACHE_DIR,
                data_dir=DATA_DIR,
            )
        )

        logging.info(f"Tokenizing dataset column {TRAIN_FEATURES_COLUMN}")
        dataset = dataset.map(
            lambda x: tokenizer(
                x[TRAIN_FEATURES_COLUMN], truncation=True, padding="max_length"
            )
        )

        logging.info(
            f"Relabeling dataset column {TRAIN_LABELS_COLUMN} using {TRAINING_RELABEL_FUNC_NAME}"
        )
        dataset = relabel_training(dataset)

        logging.info("Dropping rows in training data where label is missing")
        dataset = dataset.filter(lambda row: not (row["labels"] is None))

        logging.info("Training...")
        pretrained_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_CHECKPOINT, num_labels=NUM_LABELS
        )

        if USE_CUDA:
            pretrained_model.to("cuda")

        trainer = train(
            pretrained_model, dataset, tokenizer, RUN_OUTPUTS, AVERAGE_FSCORE_SUPPORT
        )
        trainer.save_model(args.model_output)

    if args.evaluate:
        logging.info(f"Evaluating with dataset {args.eval_dataset}")
        if args.model_input:
            eval_model = AutoModelForSequenceClassification.from_pretrained(
                args.model_input
            )
        if args.eval_dataset:
            EVALUATION_DATASET = args.eval_dataset
        if args.eval_all:
            datasets_to_eval = dataset_cols.keys()
        else:
            datasets_to_eval = [EVALUATION_DATASET]

        for dataset_name in datasets_to_eval:
            tot = loader(dataset_name, tokenizer, args.cache_dir, args.short_test)
            for eval_dataset in tot:
                torch.cuda.empty_cache()
                current_dataset = eval_dataset["train"]

                logging.info(f"Evaluating {current_dataset}")
                logging.info("Torch memory dump below")
                logging.info(pformat(torch.cuda.memory_stats(device=None)))
                tokens_tensor = current_dataset["input_ids"]
                token_type_ids = current_dataset["token_type_ids"]
                attn_masks = current_dataset["attention_mask"]
                logging.info("setting model")
                eval_model.eval()

                torch.cuda.empty_cache()

                predictions = []
                chunk = 0
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                fname_model_prefix = args.model_input.replace("/", "_")

                start_time = time.time()
                calculate_time = lambda: time.time() - start_time

                with torch.no_grad():
                    # start_time = time.time()
                    for (
                        tokens_tensor_chunk,
                        token_type_ids_chunk,
                        attn_masks_chunk,
                    ) in make_chunks(tokens_tensor, token_type_ids, attn_masks, 1000):
                        # calculate_time = lambda: time.time() - start_time

                        logging.info(f"Tokens Tensor Chunk {calculate_time()}")
                        tokens_tensor_chunk = torch.tensor(tokens_tensor_chunk)
                        token_type_ids_chunk = torch.tensor(token_type_ids_chunk)
                        attn_masks_chunk = torch.tensor(attn_masks_chunk)

                        if USE_CUDA:
                            tokens_tensor_chunk = tokens_tensor_chunk.to("cuda")
                            token_type_ids_chunk = token_type_ids_chunk.to("cuda")
                            attn_masks_chunk = attn_masks_chunk.to("cuda")

                        logging.info(f"Eval Model {calculate_time()}")
                        outputs = eval_model(
                            tokens_tensor_chunk,
                            token_type_ids=token_type_ids_chunk,
                            attention_mask=attn_masks_chunk,
                        )
                        logging.info(
                            f"Load Outputs into Predictions  {calculate_time()}"
                        )
                        predictions += outputs[0]
                        logging.info(
                            f"finished chunk {chunk} - total predictions = {len(predictions)}, writing predictions"
                        )
                    end_time = time.time()
                    logging.info(f"Time for evaluation {calculate_time()}")

                filename = f"{args.model_input}-{current_time}-{dataset_name}-train-eval.out".replace(
                    "/", "-"
                )
                with open(filename, "w") as outfile:
                    outfile.write("FULL PREDICTIONS BELOW\n")
                    outfile.write(f"args: {args}\n")
                    for text, preds in zip(current_dataset["input_text"], predictions):
                        outfile.write(f"{text} | {preds}\n")
