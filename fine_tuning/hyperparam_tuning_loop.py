# make sure optuna is installed: 
# !pip install optuna
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

metric_name = "f1"
batch_size = 8
# encoded_dataset = <should have been defined earlier in the code>
# tokenizer = <should have been defined earlier in the code>
# model_checkpoint = <should have been defined earlier in the code, e.g., 'bert-base-cased'>
# num_labels = <the number of labels in the classification task>
# model_output_dir = 

def compute_metrics(pred):
    preds, labels = pred
    preds = np.argmax(preds, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def model_init():
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    model.eval()
    model.to('cuda')
    return model


def my_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
        "seed": trial.suggest_int("seed", 1, 40),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [batch_size]), # constant batch size
    }


args = TrainingArguments(
    output_dir = model_output_dir,
    evaluation_strategy = "epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    greater_is_better=True
)


trainer = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


best_run = trainer.hyperparameter_search(n_trials=5, direction="maximize", hp_space=my_hp_space)

for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)