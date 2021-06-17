import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
import numpy as np

from array_ops import select

parser = argparse.ArgumentParser(
    description="Combine multiple ESV results into a single file for use with the ESV dashboard",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("esv_result_pickles", nargs="+", help="Path to ESV result pickle")
parser.add_argument(
    "collated_esv_results_pickle", help="Path to save collated ESV results to"
)
parser.add_argument(
    "--dataset",
    type=str,
    help="Name of dataset to put into collated file. Used in ESV dashboard",
)
parser.add_argument(
    "--model",
    type=str,
    help="Name of model to put into collated file. Used in ESV dashboard",
)

def main(args):
    result_paths = args.esv_result_pickles
    results: List[Dict[str, Any]] = [pd.read_pickle(path) for path in result_paths]
    
    narration_ids = compute_common_narration_ids(results)

    collated_results = collate_results(results, narration_ids)
    collated_results["attrs"] = {}
    if args.dataset is not None:
        collated_results["attrs"]["dataset"] = args.dataset
    if args.model is not None:
        collated_results["attrs"]["model"] = args.model

    with open(args.collated_esv_results_pickle, "wb") as f:
        pickle.dump(collated_results, f)


def compute_common_narration_ids(results: List[Dict[str, Any]]) -> np.ndarray:
    result_uids = [set(r["uids"]) for r in results]
    common_uids_set = set.union(*result_uids)
    # preserve the order of the examples based on the first result
    common_uids = np.array(
        [uid for uid in results[0]["uids"] if uid in common_uids_set]
    )
    return common_uids


def collate_results(results: List[Dict[str, Any]], uids: np.ndarray) -> Dict[str, Any]:
    first_result = results[0]
    collated_results = {
        "uids": uids,
        "labels": select(first_result["labels"], first_result["uids"], uids),
        "sequence_lengths": select(
            first_result["sequence_lengths"], first_result["uids"], uids
        )
    }

    def subsample_results(key: str) -> Union[np.ndarray, List[np.ndarray]]:
        arrays = [select(result[key], result["uids"], uids) for result in results]
        try:
            return np.stack(arrays)
        except ValueError:
            # Can't stack arrays since they aren't all the same shape
            return arrays

    for k in ["scores", "sequence_idxs", "shapley_values"]:
        collated_results[k] = subsample_results(k)
    return collated_results

if __name__ == "__main__":
    main(parser.parse_args())
