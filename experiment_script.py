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
  -tg, --training-globals    
                        A path to a file containing a TrainingGlobals object, representing global variables for model training
  -eg, --eval-globals    
                        A path to a file containing an EvalGlobals object, representing global variables for model evaluation

"""

metric_name = "f1"
AVERAGE_FSCORE_SUPPORT = "micro"
batch_size = 2
EVALUATION_DATASET = "air_dialogue"
TRAINING_DATASET = "social_bias_frames"
TRAINING_DATASET_SPLIT = None
DATASETS_DIR = "/scratch/amq259/datasets"
# MODEL_CHECKPOINT = "bert-base-cased"
MODEL_CHECKPOINT = "/scratch/pw1329/nlu-s2021-final-project/models/rtgender-model-new/"
RUN_OUTPUTS = ""
TRAIN_FEATURES_COLUMN = "post"
NUM_LABELS = 3
relabel_training = None


from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    # EarlyStoppingCallback,
)


from datasets import load_dataset, Dataset
import numpy as np
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
from relabel_funcs import relabel_social_bias_frames, relabel_rt_gender
import time

logging.basicConfig(level=logging.INFO)

USE_CUDA = False
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
    "peixian/rtGender": relabel_rt_gender,
    "mdGender": "",
    "jigsaw_toxicity_pred": lambda x: x,
    "social_bias_frames": relabel_social_bias_frames,
    "peixian/equity_evaluation_corpus": lambda x: 0 if x == "male" else 1,
}


## mapping of dataset_name to dataset columns of sentences
dataset_cols = {
    "EMBO/biolang": "special_tokens_mask",
    "empathetic_dialogues": "utterance",
    "conv_ai_3": "answer",
    "air_dialogue": "dialogue",
    "ted_talks_iwslt": "translation",
    "tweet_eval": "text",
}


dataset_types = {
    "ted_talks_iwslt": ["nl_en_2014", "nl_en_2015"],
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
}


dataset_preprocess = {
    # 'NAME: <SENTENCE>'
    # "air_dialogue": lambda x: x.split("")[-1],
    "ted_talks_iwslt": lambda x: x["en"],
}


cols_removed = {
    "air_dialogue": ["action", "correct_sample", "dialogue", "expected_action", "intent", "search_info", "timestamps"]
}

def clean_text():
    pass

def loader(dataset_name, tokenizer, cache_dir):
    assert dataset_name in dataset_cols
    sentence_col = dataset_cols[dataset_name]
    d_types = dataset_types.get(dataset_name, None)
    tot = []
    logging.info(f"Using cache dir {cache_dir}")
    if d_types:
        for d_type in d_types:
            data = load_dataset(dataset_name, d_type, cache_dir=cache_dir)
            tot.append(_preprocess_dataset(dataset_name, data, sentence_col, tokenizer))
    else:
        data = load_dataset(dataset_name, cache_dir=cache_dir)
        tot.append(_preprocess_dataset(dataset_name, data, sentence_col, tokenizer))
    return tot


def _preprocess_dataset(dataset_name, data, sentence_col, tokenizer):
    preprocess_function = dataset_preprocess.get(dataset_name, lambda x: x)
    data = data.map(lambda x: {"input_text": preprocess_function(x[sentence_col])})
    first_length = len(data['train']['input_text'])
    if dataset_name in set(["air_dialogue"]):
        for split in data:
            data[split].remove_columns(cols_removed[dataset_name])
            data[split] = Dataset.from_dict({'input_text': np.concatenate(data[split]['input_text']).ravel().tolist()})
    assert len(data['train']['input_text']) >= first_length 
    data = data.map(
        lambda x: tokenizer(x["input_text"], padding="max_length", truncation=True),
        batched=True,
    )
    return data


def compute_metrics(pred):
    preds, labels = pred
    preds = np.argmax(preds, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=AVERAGE_FSCORE_SUPPORT
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def model_init():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=4
    )
    model.train()
    if USE_CUDA:
        model.to("cuda")
    return model


def train(encoded_dataset, tokenizer):
    args = TrainingArguments(
        output_dir=RUN_OUTPUTS,
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
        model_init=model_init,
        args=args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    logging.info("Constructed trainer")

    trainer.train()
    
    return best_run, 

def make_chunks(data1, data2, chunk_size):
    while data1 and data2:
        chunk1, data1 = data1[:chunk_size], data1[chunk_size:]
        chunk2, data2 = data2[:chunk_size], data2[chunk_size:]
        yield chunk1, chunk2


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
        action="store_false",
        default=True,
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
        "-tg",
        "--training-globals",
        default="",
        required=True,
        help="A path to a file containing a TrainingGlobals object, representing global variables for model training",
    )

    parser.add_argument(
        "-eg",
        "--eval-globals",
        default="",
        required=True,
        help="A path to a file containing an EvalGlobals object, representing global variables for model evaluation",
    )

    args = parser.parse_args()
    if not args.train and not args.evaluate:
        print("Not training AND not evaluating. Exiting.")
        exit(0)
    if args.evaluate and not args.train and not args.model_input:
        print(
            "Asked for evaluation but we are not training a model or loading one. Please supply a model or train one."
        )

    USE_CUDA = args.no_cuda

    logging.info("Initializing seeds and setting values")
    logging.info(f"Use cuda? {USE_CUDA}")
    gc.collect()
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    if USE_CUDA:
        torch.cuda.empty_cache()
    batch_size = int(args.batch_size)
    logging.info(f"Using batch_size {batch_size}")
    logging.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    if args.train:
        if args.train_dataset:
            TRAINING_DATASET = args.train_dataset
        logging.info(
            f"Loading dataset {TRAINING_DATASET} with split {TRAINING_DATASET_SPLIT}"
        )
        dataset = load_dataset(
            TRAINING_DATASET, split=TRAINING_DATASET_SPLIT, cache_dir=DATASETS_DIR
        )
        logging.info(f"Relabeling dataset column {TRAIN_LABELS_COLUMN}")
        dataset = dataset.map(
            lambda x: {"labels": relabel_training(x[TRAIN_LABELS_COLUMN])}
        )
        logging.info(f"Tokenizing dataset column {TRAIN_FEATURES_COLUMN}")
        dataset = dataset.map(
            lambda x: tokenizer(
                x[TRAIN_FEATURES_COLUMN], truncation=True, padding=True
            ),
            batched=True,
        )

        logging.info("Training...")
        trainer = train(dataset, tokenizer)
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
            tot = loader(dataset_name, tokenizer, args.cache_dir)
            for eval_dataset in tot:
                for split in eval_dataset:
                    torch.cuda.empty_cache()
                    current_dataset = eval_dataset[split]

                    logging.info(f"Evaluating {current_dataset}")
                    logging.info("Torch memory dump below")
                    logging.info(pformat(torch.cuda.memory_stats(device=None)))
                    tokens_tensor = current_dataset["input_ids"]
                    token_type_ids = current_dataset["token_type_ids"]
                    logging.info("setting model")
                    eval_model.eval()

                    torch.cuda.empty_cache()
                    logging.info("evaluating")
                    predictions = []
                    chunk = 0
                    logging.info("Opening file and writing")
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    filename = f"{current_time}-{dataset_name}-{split}-partial-predictions-eval.out"
                    logging.info(f"Opening file {filename} to write results")
                    with open(filename, "w") as outfile:
                        outfile.write("PARTIAL PREDICTIONS BELOW\n")
                        outfile.write(f"args: {args}\n")
                        with torch.no_grad():
                            start_time = time.time()
                            for (
                                tokens_tensor_chunk,
                                token_type_ids_chunk,
                            ) in make_chunks(tokens_tensor, token_type_ids, 100):
                                tokens_tensor_chunk = torch.tensor(tokens_tensor_chunk)
                                token_type_ids_chunk = torch.tensor(
                                    token_type_ids_chunk
                                )
                                outputs = eval_model(
                                    tokens_tensor_chunk,
                                    token_type_ids=token_type_ids_chunk,
                                )
                                predictions += outputs[0]
                                logging.info(
                                    f"finished chunk {chunk} - total predictions = {len(predictions)}, writing predictions"
                                )
                                # output is [[123, 123, 123], [123, 123, 123]]
                                for token_ids, preds in zip(
                                    token_type_ids_chunk, outputs[0]
                                ):
                                    outfile.write(f"{tokens_tensor_chunk} | {preds}\n")

                            end_time = time.time()
                            logging.info(f"Time for evaluation {end_time - start_time}")

                    filename = f"{current_time}-{dataset_name}-{split}-eval.out"
                    with open(filename, "w") as outfile:
                        outfile.write("FULL PREDICTIONS BELOW\n")
                        outfile.write(f"args: {args}\n")
                        for text, preds in zip(
                            current_dataset["input_text"], predictions
                        ):
                            outfile.write(f"{text} | {preds}\n")
