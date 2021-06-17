from dataclasses import dataclass
from typing import Dict, List

import numpy as np

@dataclass
class Result:
    esvs: List[np.ndarray]  # [n_frames_idx][frame_idx, class_idx]
    scores: List[np.ndarray]  # [n_frames_idx, class_idx]
    uid: str
    label: Dict[str, int]
    sequence_idxs: List[np.ndarray]  # [n_frames_idx][frame_idx]
    results_idx: int

    @property
    def max_n_frames(self):
        return max([len(s) for s in self.sequence_idxs])

    @property
    def max_frame(self):
        return max([idxs[-1] for idxs in self.sequence_idxs])

class ShapleyValueResults:
    def __init__(self, results):
        self._results = results

    @property
    def uids(self) -> List[str]:
        return list(self._results["uids"])

    @property
    def shapley_values(self) -> List[np.ndarray]:
        # shapley_values[n_frames_idx][example_idx, frame_idx, class_idx]
        return self._results["shapley_values"]

    @property
    def sequence_idxs(self) -> np.ndarray:
        # sequence_idxs[n_frames_idx][example_idx]
        return self._results["sequence_idxs"]

    @property
    def labels(self) -> np.ndarray:
        return self._results["labels"]

    @property
    def scores(self) -> np.ndarray:
        # sequence_idxs[n_frames_idx, example_idx, class_idx]
        return self._results["scores"]

    @property
    def max_n_frames(self) -> int:
        return len(self._results["scores"])

    def __getitem__(self, idx: str):
        if isinstance(idx, str):
            example_idx = self.uids.index(idx)
        else:
            raise ValueError(f"Cannot handle idx type: {idx.__class__.__name__}")
            
        return Result(
            esvs=[esvs[example_idx] for esvs in self.shapley_values],
            scores=[scores[example_idx] for scores in self.scores],
            uid=self.uids[example_idx],
            label=self.labels[example_idx],
            sequence_idxs=[sequence_idxs[example_idx] for sequence_idxs in self.sequence_idxs],
            results_idx=example_idx,
        )
