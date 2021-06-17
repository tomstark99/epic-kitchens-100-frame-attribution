import argparse
import logging

from typing import Dict, Any, List, Tuple
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from torchvideo.samplers import frame_idx_to_list
from frame_sampling import RandomSampler

import torch
from datasets.pickle_dataset import MultiPickleDataset

from omegaconf import OmegaConf

from systems import EpicActionRecognitionSystem
from systems import EpicActionRecogintionDataModule
from models.esvs import V_MTRN, N_MTRN

import pickle
import numpy as np
import pandas as pd

from attribution.online_shapley_value_attributor import OnlineShapleyAttributor
from subset_samplers import ConstructiveRandomSampler

from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Compute ESVs given a trained model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("features_dir", type=Path, help="Path to features")
parser.add_argument("checkpoints", type=Path, help="Path to model checkpoints")
parser.add_argument("verb_class_priors", type=Path, help="Path to verb class priors")
parser.add_argument("noun_class_priors", type=Path, help="Path to noun class priors")
parser.add_argument("esvs_pickle", type=Path, help="Path to pickle file to save features")
parser.add_argument("--sample_n_frames", type=int, default=8, help="How many frames to sample to compute ESVs for")

def main(args):

    device = torch.device("cuda:0")
    dtype = torch.float

    # TODO: Implement
    """
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg = OmegaConf.create(ckpt['hyper_parameters'])
    OmegaConf.set_struct(cfg, False)

    cfg.data._root_gulp_dir = str(args.gulp_dir)

    model = EpicActionRecognitionSystem(cfg)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    """

    dataset = MultiPickleDataset(args.features_dir)
    dataloader = DataLoader(dataset, batch_size=1)

    n_frames = args.sample_n_frames
    frame_sampler = RandomSampler(frame_count=n_frames, snippet_length=1, test=True)

    v_models = [V_MTRN(frame_count=i) for i in range(1,9)]
    n_models = [N_MTRN(frame_count=i) for i in range(1,9)]

    assert len(v_models) == len(n_models)

    for i in range(len(v_models)):
        v_models[i].load_state_dict(torch.load(args.checkpoints / f'mtrn-frames={i+1}-type=verb.pt'))
        n_models[i].load_state_dict(torch.load(args.checkpoints / f'mtrn-frames={i+1}-type=noun.pt'))

    # model = models[n_frames]

    verb_class_priors = pd.read_csv(args.verb_class_priors, index_col='verb_class')['prior'].values
    noun_class_priors = pd.read_csv(args.noun_class_priors, index_col='noun_class')['prior'].values

    v_attributor = OnlineShapleyAttributor(
        single_scale_models=v_models,
        priors=verb_class_priors,
        n_classes=len(verb_class_priors),
        device=device,
        subset_sampler=ConstructiveRandomSampler(max_samples=128, device=device)
    )
    n_attributor = OnlineShapleyAttributor(
        single_scale_models=n_models,
        priors=noun_class_priors,
        n_classes=len(noun_class_priors),
        device=device,
        subset_sampler=ConstructiveRandomSampler(max_samples=128, device=device)
    )

    def subsample_frames(video: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor]:
        video_length = len(video)
        if video_length < n_frames:
            raise ValueError(f"Video too short to sample {n_frames} from")
        sample_idxs = np.array(frame_idx_to_list(frame_sampler.sample(video_length)))
        return sample_idxs, video[sample_idxs]

    data = {
        "labels": [],
        "uids": [],
        "sequence_idxs": [],
        "sequence_lengths": [],
        "scores": [],
        "shapley_values": [],
    }

    for i, (video, rgb_meta) in tqdm(
        enumerate(dataloader),
        unit=" action seq",
        total=len(dataloader),
        dynamic_ncols=True
    ):
        labels = {
            'verb': rgb_meta['verb_class'].item(),
            'noun': rgb_meta['noun_class'].item()
        }

        try:
            sample_idx, sample_video = subsample_frames(video.squeeze())
        except ValueError:
            print(
                f"{uid} is too short ({len(video)} frames) to sample {n_frames}"
                f"frames from."
            )
            continue


        # TODO: implement scores from model idx out of bound error
        # with torch.no_grad():
            # out = model(sample_video.to(device))

        v_esvs, v_scores = v_attributor.explain(sample_video.to(device))
        n_esvs, n_scores = n_attributor.explain(sample_video.to(device))

        verb_scores = v_scores.cpu().unsqueeze(0)
        noun_scores = n_scores.cpu().unsqueeze(0)

        result_scores = torch.cat((verb_scores, noun_scores), dim=-1)

        scores = {
            'verb': result_scores[:,:97].numpy(),#.cpu().numpy(),
            'noun': result_scores[:,97:].numpy()#.cpu().numpy()
        }

        verb_esvs = v_esvs.cpu().unsqueeze(0)
        noun_esvs = n_esvs.cpu().unsqueeze(0)

        result_esvs = torch.cat((verb_esvs, noun_esvs), dim=-1)

        esvs = {
            'verb': result_esvs[:,:,:97].numpy(),
            'noun': result_esvs[:,:,97:].numpy()
        }
        
        rgb_meta['narration_id'] = rgb_meta['narration_id'][0]

        data["labels"].append(labels)
        data["uids"].append(rgb_meta['narration_id'])
        data["sequence_idxs"].append(sample_idx)
        data["sequence_lengths"].append(video.shape[1])#rgb_meta['num_frames'])
        data["scores"].append(scores)
        data["shapley_values"].append(esvs)
    
    def collate(vs: List[Any]):
        try:
            return np.stack(vs)
        except ValueError:
            return vs

    data_to_persist = {k: collate(vs) for k, vs in data.items()}

    with open(args.esvs_pickle, 'wb') as f:
        pickle.dump(data_to_persist, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main(parser.parse_args())
