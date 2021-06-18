import argparse
import pickle
import pandas as pd

from datasets.gulp_dataset import GulpDataset
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Extract verb-noun links from a given dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("gulp_dir", type=Path, help="Path to gulp directory")
parser.add_argument("verb_noun_pickle", type=Path, help="Path to pickle file to save verb-noun links")
parser.add_argument("verb_classes", type=Path, help="Path to verb classes csv")
parser.add_argument("noun_classes", type=Path, help="Path to noun classes csv")
parser.add_argument("--classes", type=bool, default=False, help="Extract as pure class numbers")
parser.add_argument("--narration-id", type=bool, default=False, help="Extract with noun as tuple with narration_id")

def main(args):

    verbs = pd.read_csv(args.verb_classes)
    nouns = pd.read_csv(args.noun_classes)

    verb_noun = {}

    dataset = GulpDataset(args.gulp_dir)

    verb_noun = extract_verb_noun_links(
        dataset,
        verbs, 
        nouns, 
        verb_noun,
        classes=args.classes,
        narration=args.narration_id)

    with open(args.verb_noun_pickle, 'wb') as f:
        pickle.dump({
            verb: unique_list(verb_noun[verb]) for verb in verb_noun.keys()
        }, f)

def extract_verb_noun_links(
    dataset: GulpDataset,
    verbs: pd.DataFrame,
    nouns: pd.DataFrame,
    output_dict: Dict[str, List],
    classes: bool = False,
    narration: bool = False
):
    for i, (_, batch_labels) in tqdm(
        enumerate(dataset),
        unit=" action seq",
        total=len(dataset),
        dynamic_ncols=True
    ):
        if classes:
            if batch_labels['verb_class'] in output_dict:
                output_dict[batch_labels['verb_class']].append((batch_labels['noun_class'], batch_labels['narration_id']) if narration else batch_labels['noun_class'])
            else:
                output_dict[batch_labels['verb_class']] = [(batch_labels['noun_class'], batch_labels['narration_id']) if narration else batch_labels['noun_class']]
        else:
            # lookup has to be performed to get the direct verb / noun rather than the instance
            if verbs['key'][batch_labels['verb_class']] in output_dict:
                output_dict[verbs['key'][batch_labels['verb_class']]].append((nouns['key'][batch_labels['noun_class']], batch_labels['narration_id']) if narration else nouns['key'][batch_labels['noun_class']])
            else:
                output_dict[verbs['key'][batch_labels['verb_class']]] = [(nouns['key'][batch_labels['noun_class']], batch_labels['narration_id']) if narration else nouns['key'][batch_labels['noun_class']]]

    return output_dict

def unique_list(list):
    seen = set()
    return [x for x in list if not (x in seen or seen.add(x))]

# def extract_verb_noun_links(
#     dataset: GulpDirectory,
#     verbs: pd.DataFrame,
#     nouns: pd.DataFrame,
#     output_dict: Dict[str, List]
# ):
#     for i, c in tqdm(
#         enumerate(dataset),
#         unit=" chunk",
#         total=dataset.num_chunks,
#         dynamic_ncols=True
#     ):
#         for _, batch_labels in c:
#             if verbs['key'][batch_labels['verb_class']] in output_dict:
#                 output_dict[verbs['key'][batch_labels['verb_class']]].append(nouns['key'][batch_labels['noun_class']])
#             else:
#                 output_dict[verbs['key'][batch_labels['verb_class']]] = [nouns['key'][batch_labels['noun_class']]]

#     return output_dict

# def extract_verb_noun_links_classes(
#     dataset: GulpDirectory,
#     output_dict: Dict[str, List]
# ):
#     for i, c in tqdm(
#         enumerate(dataset),
#         unit=" chunk",
#         total=dataset.num_chunks,
#         dynamic_ncols=True
#     ):
#         for _, batch_labels in c:
#             if batch_labels['verb_class'] in output_dict:
#                 output_dict[batch_labels['verb_class']].append(batch_labels['noun_class'])
#             else:
#                 output_dict[batch_labels['verb_class']] = [batch_labels['noun_class']]

#     return output_dict

if __name__ == '__main__':
    main(parser.parse_args())
