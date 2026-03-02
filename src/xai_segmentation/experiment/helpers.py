import json
from typing import Any, Optional, Tuple, List

import torch

from xai_seg.xai_metrics import DropsPackFaith
import numpy as np


def drops_cfg_key(drops_cfg: dict) -> str:
    return json.dumps(drops_cfg, sort_keys=True, separators=(",", ":"))


def drops_key(
        roi_type: str, dataset_name: Any, image_id: Any, class_id: int,
        component_idx: Optional[int], H: int, W: int, drops_cfg: dict
) -> Tuple[str, Any, Any, int, Optional[int], int, int, str]:
    return (roi_type, dataset_name, image_id, int(class_id),
            (int(component_idx) if component_idx is not None else None),
            int(H), int(W), drops_cfg_key(drops_cfg))


def slice_pack_row(pack: DropsPackFaith, idx: int) -> DropsPackFaith:
    return DropsPackFaith(
        drops_all_raw=pack.drops_all_raw[idx:idx + 1].contiguous(),
        b_count_matrix=pack.b_count_matrix[idx:idx + 1].contiguous(),
    )


def stack_packs(packs: List[DropsPackFaith]) -> DropsPackFaith:
    dar = torch.cat([p.drops_all_raw for p in packs], dim=0)
    bcm = torch.cat([p.b_count_matrix for p in packs], dim=0)
    return DropsPackFaith(dar, bcm)


def get_local_and_global_masks(t: dict, ccl_cache, device):
    ccl = ccl_cache[(t["dataset_name"], t["image_id"], int(t["class_id"]))]
    m_loc  = torch.from_numpy((ccl.labels == int(t["component_idx"])).astype(np.uint8)).to(device)
    m_glob = torch.from_numpy((ccl.labels > 0).astype(np.uint8)).to(device)
    return m_loc, m_glob

def build_local_to_global_index(local_tasks, global_tasks):
    gmap = {(g["dataset_name"], g["image_id"], int(g["class_id"])): g for g in global_tasks}
    return {
        lt["mask_sig"]: {
            "local": lt,
            "global": gmap.get((lt["dataset_name"], lt["image_id"], int(lt["class_id"])))
        }
        for lt in local_tasks
    }
