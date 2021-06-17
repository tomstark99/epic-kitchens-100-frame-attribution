from typing import Dict, Any
from torch.utils.data import Dataset
from pathlib import Path

from torchvideo.samplers import FrameSampler
from torchvideo.samplers import frame_idx_to_list

import pickle
import numpy as np

class PickleDataset(Dataset):
    
    def __init__(self, pkl_path: Path, frame_sampler: FrameSampler, features_dim: int = 256):
        self.pkl_path = pkl_path
        self.frame_sampler = frame_sampler
        self.features_dim = features_dim
        self.pkl_dict = Dict[str, Any]
        self.frame_cumsum = np.array([0.])
        self._load()
        
    def _load(self):
        with open(self.pkl_path, 'rb') as f:
            self.pkl_dict = pickle.load(f)
            frame_counts = [label['num_frames'] for label in self.pkl_dict['labels']]
            self.frame_cumsum = np.cumsum(np.concatenate([self.frame_cumsum, frame_counts]), dtype=int)
    
    def _video_from_narration_id(self, key: int):
        l = self.frame_cumsum[key]
        r = self.frame_cumsum[key+1]
        return self.pkl_dict['features'][l:r]
    
    def __len__(self):
        return len(self.pkl_dict['narration_id'])
    
    def __getitem__(self, key: int):
        features = self._video_from_narration_id(key)
        video_length = features.shape[0]
        
        assert video_length == self.pkl_dict['labels'][key]['num_frames']
        if video_length < self.frame_sampler.frame_count:
            raise ValueError(f"Video too short to sample {self.frame_sampler.frame_count} from")
        
        sample_idxs = np.array(frame_idx_to_list(self.frame_sampler.sample(video_length)))
        return (features[sample_idxs], { k: self.pkl_dict['labels'][key][k] for k in ['narration_id','verb_class','noun_class'] })

class MultiPickleDataset(Dataset):
    
    def __init__(self, pkl_path: Path, features_dim: int = 256):
        self.pkl_path = pkl_path
        self.features_dim = features_dim
        self.pkl_dict = Dict[str, Any]
        self.frame_cumsum = np.array([0.])
        self._load()
        
    def _load(self):
        with open(self.pkl_path, 'rb') as f:
            self.pkl_dict = pickle.load(f)
            frame_counts = [label['num_frames'] for label in self.pkl_dict['labels']]
            self.frame_cumsum = np.cumsum(np.concatenate([self.frame_cumsum, frame_counts]), dtype=int)
    
    def _video_from_narration_id(self, key: int):
        l = self.frame_cumsum[key]
        r = self.frame_cumsum[key+1]
        return self.pkl_dict['features'][l:r]
    
    def __len__(self):
        return len(self.pkl_dict['narration_id'])
    
    def __getitem__(self, key: int):
        features = self._video_from_narration_id(key)
        video_length = features.shape[0]
        assert video_length == self.pkl_dict['labels'][key]['num_frames']
        
        return (features, { k: self.pkl_dict['labels'][key][k] for k in ['narration_id','verb_class','noun_class'] })

class TestMultiPickleDataset(Dataset):
    
    def __init__(self, pkl_path: Path, features_dim: int = 256):
        self.pkl_path = pkl_path
        self.features_dim = features_dim
        self.pkl_dict = Dict[str, Any]
        self.frame_cumsum = np.array([0.])
        self._load()
        
    def _load(self):
        with open(self.pkl_path, 'rb') as f:
            self.pkl_dict = pickle.load(f)
            frame_counts = [label['num_frames'] for label in self.pkl_dict['labels']]
            self.frame_cumsum = np.cumsum(np.concatenate([self.frame_cumsum, frame_counts]), dtype=int)
    
    def _video_from_narration_id(self, key: int):
        l = self.frame_cumsum[key]
        r = self.frame_cumsum[key+1]
        return self.pkl_dict['features'][l:r]
    
    def __len__(self):
        return len(self.pkl_dict['narration_id'])
    
    def __getitem__(self, key: int):
        features = self._video_from_narration_id(key)
        video_length = features.shape[0]
        assert video_length == self.pkl_dict['labels'][key]['num_frames']
        
        return (features, { 'narration_id': self.pkl_dict['labels'][key]['narration_id'] })
