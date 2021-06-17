from abc import ABC
from typing import Dict, Any

import numpy as np

class FeatureWriter(ABC):
    def append(self, narration_id: str, features: np.ndarray, labels: Dict[str, Any]) -> None:
        raise NotImplementedError()
