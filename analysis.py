#!/usr/bin/env python3
import argparse
import pandas as pd
import torch.tensor as pt_tensor

def tensor_split_mapper(input_tensor):
    # for number in [123, 123, 123]:
    # create column named prediction0, prediction1, prediction2
    # new columns names = range(len(eval(tensor string)))
    # apply for all N

    # return a dictionary for each row
    input_parsed_tensor = input_tensor
    prediction_counts = len(input_parsed_tensor)
    out = {}
    for prediction_num in range(0, prediction_counts):
        out[f"prediction{prediction_num}"] = input_parsed_tensor[prediction_num]
    return out


def read_outfile(outfile_name, delimiter="|", skiprows=2, split_tensor=True):
    df = pd.read_csv(
        outfile_name,
        delimiter=delimiter,
        skiprows=skiprows,
        names=["sentence", "predictions"],
    )
    df["predictions"] = df["predictions"].apply(
        lambda x: x.replace("tensor(", "").replace(")", "")
    )
    df["predictions"] = df["predictions"].map(lambda x: pt_tensor(eval(x)))
    if split_tensor:
        df = df.join(
            pd.DataFrame(df["predictions"].apply(tensor_split_mapper).tolist())
        )
    return df


def read_dfs(file1, file2, suffixes=("_df1", "_df2"), split_tensor=False):
    df1 = read_outfile(file1, split_tensor=split_tensor)
    df2 = read_outfile(file2, split_tensor=split_tensor)
    combined_df = df1.merge(df2, on="sentence", suffixes=suffixes)
    return df1, df2, combined_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i1", "--input1", required=True, help="First input file")
    parser.add_argument(
        "-i2",
        "--input2",
        help="""Second input file. Optional. If given, runs the correlation between the two files. """,
        default=None,
    )

    args = parser.parse_args()
    df1 = read_outfile(args.input1)
    print(df1.describe())
    if args.input2:
        df2 = read_outfile(args.input2)
        print(df2.describe())
        combined_df = df1.merge(df2, on="sentence", suffixes=("_df1", "_df2"))
        print(combined_df.corr())
