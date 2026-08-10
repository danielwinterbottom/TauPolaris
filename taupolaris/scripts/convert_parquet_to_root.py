#!/usr/bin/env python3

from pathlib import Path
import argparse
import pandas as pd
import uproot


def parquet_to_root(input_parquet: str):
    input_path = Path(input_parquet)
    output_root = input_path.with_suffix(".root")

    df = pd.read_parquet(input_path)

    with uproot.recreate(output_root) as root_file:
        root_file["tree"] = df

    print(f"Wrote TTree 'tree' to {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a Parquet dataframe to a ROOT file containing a TTree named 'tree'."
    )
    parser.add_argument("input_parquet", help="Input Parquet file")

    args = parser.parse_args()
    parquet_to_root(args.input_parquet)
