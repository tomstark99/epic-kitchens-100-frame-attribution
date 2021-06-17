import argparse
import logging

from gulpio2 import GulpDirectory
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List

import torch
import torch.multiprocessing
from torch.utils.data import Subset

from omegaconf import OmegaConf

from systems import EpicActionRecognitionSystem
from systems import EpicActionRecogintionDataModule

from features.feature_extractor import FeatureExtractor
from features.pkl import PickleFeatureWriter
from datasets.gulp_dataset import GulpDataset

from ipdb import launch_ipdb_on_exception

parser = argparse.ArgumentParser(
    description="Extract per-frame features from given dataset and backbone",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("gulp_dir", type=Path, help="Path to gulp directory")
parser.add_argument("checkpoint", type=Path, help="Path to model checkpoint")
parser.add_argument("features_pickle", type=Path, help="Path to pickle file to save features")
parser.add_argument("--num_workers", type=int, default=0, help="Number of features expected from frame")
parser.add_argument("--batch_size", type=int, default=128, help="Max frames to run through backbone 2D CNN at a time")
parser.add_argument("--feature_dim", type=int, default=256, help="Number of features expected from frame")

def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float

    torch.multiprocessing.set_sharing_strategy('file_system')

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg = OmegaConf.create(ckpt['hyper_parameters'])
    OmegaConf.set_struct(cfg, False)

    cfg.data._root_gulp_dir = str(args.gulp_dir)

    model = EpicActionRecognitionSystem(cfg)
    model.load_state_dict(ckpt['state_dict'])
    
    dataset = GulpDataset(args.gulp_dir)
    feature_writer = PickleFeatureWriter(args.features_pickle, features_dim=args.feature_dim)
    dataset_subsample = Subset(dataset, torch.arange(feature_writer.length, len(dataset)))

    extractor = FeatureExtractor(model.model.to(device), device, dtype, dataset_subsample, frame_batch_size=args.batch_size)
    with launch_ipdb_on_exception():
        total_instances = extract_features_to_pkl(
            extractor, feature_writer, args.feature_dim, args.num_workers
        )

    print(f"extracted {total_instances} features.")

def extract_features_to_pkl(
    feature_extractor: FeatureExtractor,
    feature_writer: PickleFeatureWriter,
    feature_dim: int,
    num_workers: int
):
    total_instances = 0
    
    dataloader = feature_extractor.get_dataloader(num_workers=num_workers)
    

    total_instances += feature_extractor.extract(dataloader, feature_writer)
    feature_writer.save()

    return total_instances

if __name__ == "__main__":
    main(parser.parse_args())

