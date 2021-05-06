import datasets
from datasets import DatasetDict


def relabel_md_gender_yelp(dataset):

    dataset = dataset.rename_column("binary_label", "labels")

    return dataset

def relabel_md_gender_wizard(dataset):

    dataset = dataset.rename_column("gender", "labels")

    return dataset


def relabel_md_gender_convai_binary(dataset):

    dataset = dataset.rename_column("binary_label", "labels")

    return dataset


def relabel_md_gender_convai_ternary(dataset):

    dataset = dataset.rename_column("ternary_label", "labels")

    return dataset

def split_relabel_eec(dataset):

    def relabel_func(column):

        relabel_dict = {
            '': 0,
            'anger': 1,
            'fear': 2,
            'joy': 3,
            'sadness': 4
        }

        return [relabel_dict[elt] for elt in column]

    dataset = dataset.map(lambda x: {'labels': relabel_func(x['emotion'])},  batched=True)
    train_test = dataset['train'].train_test_split(test_size=0.20)
    train_val = train_test['train'].train_test_split(test_size=0.25)
    dataset = DatasetDict({
        'train': train_val['train'],
        'test': train_test['test'],
        'validation': train_val['test']}
    )

    return dataset


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