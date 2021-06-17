import argparse
from pathlib import Path
import pandas as pd
import pickle

parser = argparse.ArgumentParser(
    description="Compute epic-100 noun class priors from empirical class frequency",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("train_pkl", type=Path, help="Path JSON file containing array of")
parser.add_argument(
    "class_priors_csv", type=Path, help="Path to CSV file to save class priors to."
)

def main(args):

    with open(args.train_pkl, 'rb') as f:
        train_labels = pickle.load(f)
    # train_labels = pd.read_pickle(args.train_pkl)

    noun_class_frequencies = train_labels['noun_class'].value_counts().sort_index()

    noun_class_priors = noun_class_frequencies / noun_class_frequencies.sum()
    noun_class_priors.index.name = 'noun_class'
    noun_class_priors.name = 'prior'
    
    noun_class_priors_index_fix = noun_class_priors.reindex(list(range(0,300)), fill_value=0)

    noun_class_priors_index_fix.to_csv(args.class_priors_csv)


if __name__ == '__main__':
    main(parser.parse_args())
