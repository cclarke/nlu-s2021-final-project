#!/usr/bin/env python3
import argparse
import pandas as pd
import torch.tensor as pt_tensor

import pandas as pd
import analysis_relabel_funcs
from analysis import read_dfs
import torch
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency

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
    parser.add_argument("-o", "--output-filename", required=True, help="""Output file name """)

    args = parser.parse_args()

    filename = args.output_filename

    with open(filename, "w") as output_file:

        df1 = read_outfile(args.input1)
        df1_describe = df1.describe()

        output_file.write(f"Description of data at {args.input1}\n{df1_describe}\n\n")

        if args.input2:
            df2 = read_outfile(args.input2)
            df2_describe = df2.describe()
            output_file.write(f"Description of data at {args.input2}\n{df2_describe}\n\n")

            combined_df = df1.merge(df2, on="sentence", suffixes=("_df1", "_df2"))
            corr_df = combined_df.corr()
            output_file.write(f"Correlations:\n{corr_df}")
