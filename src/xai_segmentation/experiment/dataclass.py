from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Any

import numpy as np
import torch


@dataclass
class CCLItem:
    labels: np.ndarray
    sizes: np.ndarray
    H: int
    W: int
    pad_info: Optional[dict] = None


CCLCache = Dict[Tuple[Any, Any, int], CCLItem]
