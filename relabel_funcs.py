from torch.nn import Softmax
import datasets
from datasets import DatasetDict


def _softmax_and_relabel(predictions, categorical_labels):
    # takes in a torch.Tensor of predictions and a list of categorical labels, returns a tuple of (softmax tensor, categorical label)
    m = Softmax(dim=0)
    sm = m(predictions)
    return sm, categorical_labels[sm.argmax().item()]


def split_relabel_jigsaw_toxic(dataset):

    dataset = dataset.rename_column("toxic", "labels")
    train_val = dataset['train'].train_test_split(test_size=0.25)
    dataset = DatasetDict({
        'train': train_val['train'],
        'test': dataset['test'],
        'validation': train_val['test']}
    )

    return dataset

def split_relabel_jigsaw_severetoxic(dataset):

    dataset = dataset.rename_column("severe_toxic", "labels")
    train_val = dataset['train'].train_test_split(test_size=0.25)
    dataset = DatasetDict({
        'train': train_val['train'],
        'test': dataset['test'],
        'validation': train_val['test']}
    )

    return dataset

def split_relabel_jigsaw_identityhate(dataset):

    dataset = dataset.rename_column("identity_hate", "labels")
    train_val = dataset['train'].train_test_split(test_size=0.25)
    dataset = DatasetDict({
        'train': train_val['train'],
        'test': dataset['test'],
        'validation': train_val['test']}
    )

    return dataset




# -----------------------Social Bias Frames-------------------

# Each dataset should have two functions, one that relabels the dataset, and another that turns the vector into a single score
def relabel_sbic_offensiveness(dataset):

    def relabel_func(column):

        relabel_dict = {
            '0.0': 0, # not offensive
            '0.5': 1, # maybe offiensive
            '1.0': 2, # offensive
            '': None # missing value
        }

        return [relabel_dict[elt] for elt in column]

    dataset = dataset.map(lambda x: {'labels': relabel_func(x['offensiveYN'])},  batched=True)

    new_features = dataset['train'].features.copy()
    new_features["labels"] = datasets.ClassLabel(names=['no', 'maybe', 'yes'])

    dataset['train'] = dataset['train'].cast(new_features)
    dataset['validation'] = dataset['validation'].cast(new_features)
    dataset['test'] = dataset['test'].cast(new_features)

    return dataset

def filter_relabel_sbic_targetcategory(dataset):

    def relabel_func(column):

        relabel_dict = {
            '': 0, # no target category, but still offensive or maybe offensive (since we're filtering out non-offensive rows)
            'body': 1, 
            'culture': 2, 
            'disabled': 3, 
            'gender': 4, 
            'race': 5, 
            'social': 6, 
            'victim': 7,
        }

        return [relabel_dict[elt] for elt in column]

    # Filter out rows where at least some individual or group is the target of the offensive speech 
    dataset = dataset.filter(lambda row: not (row['whoTarget'] == ''))

    # relabel targetCategory
    dataset = dataset.map(lambda x: {'labels': relabel_func(x['targetCategory'])},  batched=True)

    new_features = dataset['train'].features.copy()
    new_features["labels"] = datasets.ClassLabel(names=[
        'none','body', 'culture', 'disabled', 'gender', 'race', 'social', 'victim'
        ])

    dataset['train'] = dataset['train'].cast(new_features)
    dataset['validation'] = dataset['validation'].cast(new_features)
    dataset['test'] = dataset['test'].cast(new_features)

    return dataset


def return_social_bias_frames(predictions):
    # predictions for social bias frames is a 3 size vector with [not offensive, maybe offensive, offensive]
    # returns a tuple of a score and a categorical variable
    categorical_labels = ["Not", "Maybe", "Yes"]
    softmax_preds, category = _softmax_and_relabel(predictions, categorical_labels)
    # if maybe has the highest value, score is 0
    # otherwise, take the difference between Yes and No
    if category == "Maybe":
        return 0, category
    else:
        return softmax_preds[2] - softmax_preds[0], category


# -----------------------rtGender-------------------

# Each dataset should have two functions, one that relabels the dataset, and another that turns the vector into a single score
def split_relabel_rt_gender(dataset):

    def relabel_func(column):

        relabel_dict = {
            'M': 0,
            'W': 1
        }

        return [relabel_dict[elt] for elt in column]

    dataset = dataset.map(lambda x: {'labels': relabel_func(x['op_gender'])},  batched=True)
    train_test = dataset['train'].train_test_split(test_size=0.20)
    train_val = train_test['train'].train_test_split(test_size=0.25)
    dataset = DatasetDict({
        'train': train_val['train'],
        'test': train_test['test'],
        'validation': train_val['test']}
    )

    return dataset



def return_rt_gender(predictions):
    # predictions for social bias frames is a 3 size vector with [not offensive, maybe offensive, offensive]
    # returns a tuple of a score and a categorical variable
    categorical_labels = ["Negative", "Neutral", "Mixed", "Positive"]
    softmax_preds, category = _softmax_and_relabel(predictions, categorical_labels)
    # if maybe has the highest value, score is 0
    # otherwise, take the difference between Yes and No
    if category == "Maybe":
        return 0, category
    else:
        return softmax_preds[2] - softmax_preds[0], category
