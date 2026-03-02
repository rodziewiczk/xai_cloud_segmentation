import glob
import os
import random
from typing import Optional

import numpy as np
import torch


def load_cloudsen_npz(path):
    with np.load(path) as npz:
        return npz['x']


def iter_cloudsen_from_npz(base_dir, start_idx: int = 0, limit: Optional[int] = None, pad: int = 5):
    i, n_done = start_idx, 0
    while True:
        filename = f"{i:0{pad}d}.npz"
        path = os.path.join(base_dir, filename)
        if not os.path.isfile(path):
            break
        X = torch.from_numpy(load_cloudsen_npz(path)).float()
        meta = {"dataset_name": "cloudsen", "image_id": i, "path": os.path.abspath(path)}
        yield X, meta
        i += 1
        n_done += 1
        if limit is not None and n_done >= limit:
            break


def iter_ftw_from_pt(dir_with_pt, limit=None, seed=1337):
    files = sorted(glob.glob(os.path.join(dir_with_pt, "**", "*.pt"), recursive=True))
    if seed is not None:
        random.Random(seed).shuffle(files)
    if limit is not None:
        files = files[:limit]
    for i, p in enumerate(files, 1):
        d = torch.load(p, map_location="cpu")
        X = d["image"].float()
        meta = {"dataset_name": "ftw", "image_id": i, "path": os.path.abspath(p)}
        yield X, meta
