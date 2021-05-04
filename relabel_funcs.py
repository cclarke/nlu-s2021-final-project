from torch.nn import Softmax
import datasets


def _softmax_and_relabel(predictions, categorical_labels):
    # takes in a torch.Tensor of predictions and a list of categorical labels, returns a tuple of (softmax tensor, categorical label)
    m = Softmax(dim=0)
    sm = m(predictions)
    return sm, categorical_labels[sm.argmax().item()]


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

def relabel_sbic_targetcategory(dataset):

    def relabel_func(column):

        relabel_dict = {
            'body': 0, 
            'culture': 1, 
            'disabled': 2, 
            'gender': 3, 
            'race': 4, 
            'social': 5, 
            'victim': 6,
            '': None # missing value
        }

        return [relabel_dict[elt] for elt in column]

    dataset = dataset.map(lambda x: {'labels': relabel_func(x['targetCategory'])},  batched=True)

    new_features = dataset['train'].features.copy()
    new_features["labels"] = datasets.ClassLabel(names=['no', 'maybe', 'yes'])

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
def relabel_rt_gender(toxicity):
    # rtGender has 4 types: ['Negative', 'Positive', 'Neutral', 'Mixed']
    if sentiment == "Negative":
        return 0
    elif sentiment == "Neutral":
        return 1
    elif sentiment == "Mixed":
        return 2
    else:
        return 3


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
