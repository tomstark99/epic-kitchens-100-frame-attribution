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

    verb_class_frequencies = train_labels['verb_class'].value_counts().sort_index()

    verb_class_priors = verb_class_frequencies / verb_class_frequencies.sum()
    verb_class_priors.index.name = 'verb_class'
    verb_class_priors.name = 'prior'
    
    verb_class_priors_index_fix = verb_class_priors.reindex(list(range(0,97)), fill_value=0)

    verb_class_priors_index_fix.to_csv(args.class_priors_csv)


if __name__ == '__main__':
    main(parser.parse_args())
