#!/usr/bin/env python3
import argparse
import pandas as pd
import torch.tensor as pt_tensor

import pandas as pd
from analysis_relabel_funcs import (
    return_social_bias_frames_offensiveness,
    return_rt_gender,
    return_jigsaw_toxicity,
    return_mdgender_convai_binary
)

import torch
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency


analysis_relabel_functions = {
    "return_social_bias_frames_offensiveness": return_social_bias_frames_offensiveness,
    "return_rt_gender": return_rt_gender,
    "return_jigsaw_toxicity": return_jigsaw_toxicity,
    "return_mdgender_convai_binary": return_mdgender_convai_binary
}


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
    df = df.dropna()
    df = df[df["sentence"].str.len() > 20]
    try:
        df["predictions"] = df["predictions"].apply(
            lambda x: x.replace("tensor(", "").replace(")", "").strip()
        )
    except Exception as e:
        print(f"attempted to read and failed {outfile_name}")
        raise e
    df = df[df["predictions"].str.startswith("[")]
    df["predictions"] = df["predictions"].map(lambda x: pt_tensor(eval(x)))
    if split_tensor:
        df = df.join(
            pd.DataFrame(df["predictions"].apply(tensor_split_mapper).tolist())
        )
    return df


def read_dfs(file1, file2, suffixes=("_df1", "_df2"), split_tensor=False):
    df1 = read_outfile(file1, split_tensor=split_tensor)
    df1.drop_duplicates(subset="sentence", inplace=True)
    
    df2 = read_outfile(file2, split_tensor=split_tensor)
    df2.drop_duplicates(subset="sentence", inplace=True)

    combined_df = df1.merge(df2, on="sentence", suffixes=suffixes, validate="one_to_one")
    return df1, df2, combined_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i1", "--input1", required=True, help="First input file")
    parser.add_argument("-r1", "--relabel1", required=True, help="Relableing function for first input file")
    parser.add_argument(
        "-i2",
        "--input2",
        help="""Second input file. Optional. If given, runs the correlation between the two files. """,
        default=None,
    )
    parser.add_argument("-r2", "--relabel2", help="Relableing function for second input file", default=None)
    parser.add_argument("-o", "--output-filename", required=True, help="""Output file name """)

    args = parser.parse_args()

    filename = args.output_filename

    with open(filename, "w") as output_file:

        df1 = read_outfile(args.input1)
        rfunc_1 = analysis_relabel_functions[args.relabel1]

        df1_temp = df1["predictions"].map(rfunc_1)
        df1_temp = pd.DataFrame(df1_temp.to_list(), columns=["scores_1", "category_1"])
        df1_temp["scores_1"] = df1_temp["scores_1"].map(lambda x: x.item())
        df1 = df1.join(df1_temp)

        df1_describe = df1["scores_1"].describe()

        output_file.write(f"Description of scores (unscaled) for data at {args.input1}\n{df1_describe}\n\n")

        if args.input2:
            df2 = read_outfile(args.input2)
            rfunc_2 = analysis_relabel_functions[args.relabel2]

            df2_temp = df2["predictions"].map(rfunc_2)
            df2_temp = pd.DataFrame(df2_temp.to_list(), columns=["scores_2", "category_2"])
            df2_temp["scores_2"] = df2_temp["scores_2"].map(lambda x: x.item())
            df2 = df2.join(df2_temp)

            df2_describe = df2["scores_2"].describe()

            output_file.write(f"Description of scores (unscaled) for data at {args.input2}\n{df2_describe}\n\n")

            combined_df = df1.merge(df2, on="sentence", suffixes=("_df1", "_df2"))

            std_scaler = StandardScaler()
            scores_df = combined_df[["scores_1", "scores_2"]]
            scores_df = pd.DataFrame(std_scaler.fit_transform(scores_df), columns=scores_df.columns)

            corr_df = scores_df.corr()
            output_file.write(f"Correlations:\n{corr_df}\n\n")

            contingency = pd.crosstab(combined_df["category_1"], combined_df["category_2"])
            output_file.write(f"Contingency matrix:\n{contingency}\n\n")

            _, p, _, _ = chi2_contingency(contingency)
            output_file.write(f"p-value from a Chi^2 test is:\n{p}")

