from torch.nn import Softmax

def _softmax_and_relabel(predictions, categorical_labels):
    # takes in a torch.Tensor of predictions and a list of categorical labels, returns a tuple of (softmax tensor, categorical label)
    m = Softmax(dim=0)
    sm = m(predictions)
    return sm, categorical_labels[sm.argmax().item()]


def return_social_bias_frames_offensiveness(predictions):
    # predictions for social bias frames is a 3 size vector with [not offensive, maybe offensive, offensive]
    # returns a tuple of a score and a categorical variable
    categorical_labels = ["Not Offensive", "Maybe Offensive", "Offensive"]
    softmax_preds, category = _softmax_and_relabel(predictions, categorical_labels)
    # if maybe has the highest value, score is 0
    # otherwise, take the difference between Yes and No
    if category == "Maybe Offensive":
        return 0, category
    else:
        return softmax_preds[2] - softmax_preds[0], category

def return_rt_gender(predictions):

    categorical_labels = ["Man", "Woman"]
    softmax_preds, category = _softmax_and_relabel(predictions, categorical_labels)

    return softmax_preds[1] - softmax_preds[0], category

def return_jigsaw_toxicity(predictions):

    categorical_labels = ["Not Toxic", "Toxic"]
    softmax_preds, category = _softmax_and_relabel(predictions, categorical_labels)

    return softmax_preds[1] - softmax_preds[0], category


def return_mdgender_convai_binary(predictions):

    categorical_labels = ["Female", "Male"]
    softmax_preds, category = _softmax_and_relabel(predictions, categorical_labels)

    return softmax_preds[1] - softmax_preds[0], category


def return_mdgender_convai_ternary(predictions):
    
    categorical_labels = ["female", "male", "neutral"]
    softmax_preds, category = _softmax_and_relabel(predictions, categorical_labels)
    
    if category == "neutral":
        return 0, category
    else:
        return softmax_preds[1] - softmax_preds[0], category


def return_mdgender_wizard(predictions):

    categorical_labels = ["neutral", "female", "male"]
    softmax_preds, category = _softmax_and_relabel(predictions, categorical_labels)

    if category == "neutral":
        return 0, category
    else:
        return softmax_preds[2] - softmax_preds[1], category



'BERT-SBIC-targetcategory': ['none', 'body', 'culture', 'disabled', 'gender', 'race', 'social', 'victim'], # not used
'BERT-eec-emotion': ['none', 'anger','fear', 'joy', 'sadness'], # not used













