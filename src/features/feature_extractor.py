from typing import Dict, Any
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

import numpy as np
import torch as t
import torch.nn as nn

from .pkl import PickleFeatureWriter

class FeatureExtractor:
    """
    Extracts image features from a 2D CNN backbone for every frame in all videos.
    """
    
    def __init__(
        self, 
        backbone_2d: nn.Module, 
        device: t.device, 
        dtype: t.float, 
        dataset: Dataset,
        frame_batch_size: int = 128
    ):
        self.model = backbone_2d
        self.device = device
        self.dtype = dtype
        self.dataset = dataset
        self.frame_batch_size = frame_batch_size
        self.dataloader_batch_size = 1

    def get_dataloader(self, num_workers: int = 4):
        return DataLoader(self.dataset, batch_size=self.dataloader_batch_size, num_workers=num_workers)
    
    def extract(self, dataloader: DataLoader, feature_writer: PickleFeatureWriter) -> int:
        total_instances = 0
        self.model.eval()
        for i, (batch_input, batch_labels) in tqdm(
            enumerate(dataloader),
            unit=" action seq",
            total=len(dataloader),
            dynamic_ncols=True
        ):
        
            batch_input = t.cat(batch_input).to(dtype=self.dtype)
            batch_input = batch_input.permute(0,3,1,2)
            batch_input = batch_input.unsqueeze(0)

            batch_size, n_frames = batch_input.shape[:2]
            flattened_batch_input = batch_input.view((-1, *batch_input.shape[2:]))

            n_chunks = int(np.ceil(len(flattened_batch_input)/self.frame_batch_size))
            chunks = t.chunk(flattened_batch_input, n_chunks, dim=0)
            
            flatten_batch_features = []
            for chunk in chunks:
                chunk = chunk.unsqueeze(0)
                with t.no_grad():
                    chunk_features = self.model.features(chunk.to(self.device))
                    chunk_features = self.model.new_fc(chunk_features)
                    flatten_batch_features.append(chunk_features.squeeze(0))
            flatten_batch_features = t.cat(flatten_batch_features, dim=0)
            batch_features = flatten_batch_features.view((batch_size, 
                                                        n_frames, 
                                                        *flatten_batch_features.shape[1:]))

            total_instances += batch_size
            self._append(batch_features, batch_labels, batch_size, feature_writer)
            
        return total_instances

    def _append(self, batch_features, batch_labels, batch_size, feature_writer):
        batch_narration_id = batch_labels['narration_id']
        assert batch_size == self.dataloader_batch_size
        assert len(batch_narration_id) == batch_size
        assert len([batch_labels]) == batch_size
        assert len(batch_features) == batch_size
        batch_features = batch_features.squeeze(0).cpu().numpy()

        feature_writer.append(batch_narration_id, batch_features, batch_labels)

