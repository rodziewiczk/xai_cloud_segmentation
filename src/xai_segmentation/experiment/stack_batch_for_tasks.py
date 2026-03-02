from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from experiment.dataclass import CCLCache


def stack_batch_for_tasks(
        tasks: List[dict],
        *,
        image_bank: Dict[Any, Tuple[torch.Tensor, dict]],
        ccl_cache: CCLCache,
        device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    xs: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    H = W = None

    for t in tasks:
        iid = t["image_id"]
        dname = t["dataset_name"]
        cid = int(t["class_id"])

        x_norm, _ = image_bank[iid]
        xs.append(x_norm)

        ccl = ccl_cache[(dname, iid, cid)]
        H, W = int(ccl.H), int(ccl.W)

        if t["roi_type"] == "local":
            comp_idx = int(t["component_idx"])
            m = (ccl.labels == comp_idx)
        else:
            m = (ccl.labels > 0)

        masks.append(torch.from_numpy(m.astype(np.uint8)))

    X_batch = torch.stack(xs, 0).to(device)  # (B,C,H,W)
    mask_batch = torch.stack(masks, 0).to(device=device, dtype=torch.bool)  # (B,H,W)
    return X_batch, mask_batch, H, W
