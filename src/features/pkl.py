from typing import Dict, Any
from .feature_store import FeatureWriter
from pathlib import Path

import pickle
import numpy as np

class PickleFeatureWriter(FeatureWriter):
    
    def __init__(self, pkl_path: Path, features_dim: int):
        self.pkl_path = pkl_path
        self.features_dim = features_dim
        self.length = 0
        self.narration_ids = []
        self.features = []
        self.labels = []
        self.load()
        
    def append(self, narration_id: str, features: np.ndarray, labels: Dict[str, Any]) -> None:
        assert features.shape[1] == self.features_dim
        self.narration_ids.append(narration_id[0] if isinstance(narration_id, list) else narration_id)
        self.features.append(features)
        self.labels.append(labels)
        self.length = len(self.narration_ids)
        try:
            self.save()
        except Exception as e:
            print(e)
        
    def save(self):
        # self.chunk_no = chunk_no
        for i, label in enumerate(self.labels):
            for k, v in label.items():
                if isinstance(label[k], list):
                    try:
                        self.labels[i][k] = v[0]
                    except Exception:
                        pass

        with open(self.pkl_path, 'wb') as f:
            pickle.dump({
                'length': self.length,
                'narration_id': self.narration_ids,
                'features': np.concatenate(self.features),
                'labels': self.labels
            }, f)
    
    def load(self):
        try:
            with open(self.pkl_path, 'rb') as f:
                pkl_dict = pickle.load(f)
                self.length = pkl_dict['length']
                self.narration_ids = pkl_dict['narration_id']
                self.features = [pkl_dict['features']]
                self.labels = pkl_dict['labels']
        except FileNotFoundError:
            pass
