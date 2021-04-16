#!/usr/bin/env python3


metric_name = "f1"
batch_size = 8

EVALUATION_DATASET = "ted_talks_iwslt"
TRAINING_DATASET = ""
TRAINING_DATASET_SPLIT = ""
DATASETS_DIR = None

MODEL_CHECKPOINT = "bert-base-cased"
RUN_OUTPUTS = "./runs"
TRAIN_LABELS_COLUMN = ""
TRAIN_FEATURES_COLUMN = ""
NUM_LABELS = 3

relabel_training = None


def relabel_training(offensive):
    if offensive:
        if offensive == "0.0":
            return 0
        elif offensive == "0.5":
            return 1
        else:
            return 2
    else:
        return 0


from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import datasets
from datasets import load_dataset, load_metric
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
)

#torch.cuda.memory_summary(device=None, abbreviated=False)
import gc
import random
import argparse
import logging
from datetime import datetime

gc.collect()
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.cuda.empty_cache()

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
    "air_dialogue": lambda x: x.split("")[-1],
    "ted_talks_iwslt": lambda x: x["en"],
}


def loader(dataset_name, tokenizer):
    assert dataset_name in dataset_cols
    sentence_col = dataset_cols[dataset_name]
    d_types = dataset_types.get(dataset_name, [])

    tot = []

    for d_type in d_types:
        data = load_dataset(dataset_name, d_type)
        preprocess_function = dataset_preprocess.get(dataset_name, lambda x: x)

        data = data.map(lambda x: {sentence_col: preprocess_function(x[sentence_col])})
        data = data.map(
            lambda x: tokenizer(x[sentence_col], truncation=True, padding=True),
            batched=True,
        )
        tot.append(data)

    return tot


def compute_metrics(pred):
    preds, labels = pred
    preds = np.argmax(preds, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def model_init():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=NUM_LABELS
    )
    model.eval()
    model.to("cuda")
    return model


def my_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
        "seed": trial.suggest_int("seed", 1, 40),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [batch_size]
        ),  # constant batch size
    }


def train(encoded_dataset, tokenizer):

    args = TrainingArguments(
        output_dir=RUN_OUTPUTS,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
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

    best_run = trainer.hyperparameter_search(
        n_trials=5, direction="maximize", hp_space=my_hp_space
    )

    for n, v in best_run.hyperparameters.items():
        setattr(trainer.args, n, v)

    return best_run, trainer


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
        "-e",
        "--evaluate",
        action="store_true",
        default=False,
        help="Run an evaluation (either requires --train or a model_input)",
    )
    parser.add_argument(
        "-mi",
        "--model_input",
        default="",
        help="Where to load a model (for evaluation purposes)",
    )

    args = parser.parse_args()

    if not args.train and not args.evaluate:
        print("Not training AND not evaluating. Exiting.")
        exit(0)

    if args.evaluate and not args.train and not args.model_input:
        print(
            "Asked for evaluation but we are not training a model or loading one. Please supply a model or train one."
        )
        exit(0)

    logging.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)

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
        lambda x: tokenizer(x[TRAIN_FEATURES_COLUMN], truncation=True, padding=True),
        batched=True,
    )

    if args.train:
        logging.info("Training...")
        best_run, trainer = train(dataset, tokenizer)
        trainer.save_model(args.model_output)

    if args.evaluate:
        logging.info(f"Evaluating with dataset {EVALUATION_DATASET}")
        if args.model_input:
            eval_model = AutoModelForSequenceClassification.from_pretrained(
                args.model_input
            )
            trainer = Trainer(model=eval_model)
        tot = loader(EVALUATION_DATASET, tokenizer)

        for eval_dataset in tot:
            for split in eval_dataset:
                logging.info(f"Evaluating {split}")
                results = trainer.predict(split)

                logging.info(f"Metrics: {results.metrics}")
                now = datetime.now()

                current_time = now.strftime("%H:%M:%S")
                filename = f"{current_time}-eval.out"
                np.savetxt(filename, results.predictions)
