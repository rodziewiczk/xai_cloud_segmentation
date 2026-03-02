from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from experiment.dataclass import CCLCache
from xai_seg.xai_metrics import calculate_drops_batch, DropsPackFaith
from experiment.helpers import drops_cfg_key, slice_pack_row
from experiment.precompute_ccl_cache import DropsCache
from experiment.stack_batch_for_tasks import stack_batch_for_tasks



def precompute_all_drops_to_ram(
        model: nn.Module,
        *,
        local_tasks: List[dict],
        global_tasks: List[dict],
        image_bank: Dict[Any, Tuple[torch.Tensor, dict]],
        ccl_cache: CCLCache,
        device: torch.device,
        drops_cfg: dict,
        drops_masks_batch_size: int,
        gc_every: int = 0,
) -> DropsCache:
    drops_cache: DropsCache = {}
    sig_to_pack: Dict[str, DropsPackFaith] = {}
    all_tasks = (local_tasks or []) + (global_tasks or [])
    cfg_key_str = drops_cfg_key(drops_cfg)

    def _group_by_class(tasks: List[dict]) -> Dict[int, List[dict]]:
        out: Dict[int, List[dict]] = {}
        for t in tasks:
            out.setdefault(int(t["class_id"]), []).append(t)
        return out

    chunk_counter = 0
    for cid, tasks in _group_by_class(all_tasks).items():
        for start in range(0, len(tasks), drops_masks_batch_size):
            chunk = tasks[start:start + drops_masks_batch_size]

            Xb, Mb, H, W = stack_batch_for_tasks(
                chunk, image_bank=image_bank, ccl_cache=ccl_cache, device=device
            )
            Mb = (Mb > 0)

            chunk_sigs = [t.get("mask_sig") for t in chunk]
            new_idx = [i for i, s in enumerate(chunk_sigs) if s not in sig_to_pack]

            old_tc = getattr(model, "target_class", None)
            old_mask = getattr(model, "mask", None)
            try:
                setattr(model, "target_class", int(cid))

                if new_idx:
                    Xb_new = Xb[new_idx]
                    Mb_new = Mb[new_idx]
                    setattr(model, "mask", Mb_new)

                    dar, bcm = calculate_drops_batch(model, Xb_new, **drops_cfg)
                    partial = DropsPackFaith(dar, bcm)

                    for k, idx_in_chunk in enumerate(new_idx):
                        sig = chunk_sigs[idx_in_chunk]
                        sig_to_pack[sig] = slice_pack_row(partial, k)

                for j, t in enumerate(chunk):
                    key = (
                        t["roi_type"], t["dataset_name"], t["image_id"], int(cid),
                        (int(t.get("component_idx")) if t.get("component_idx") is not None else None),
                        int(H), int(W), cfg_key_str
                    )
                    pack = sig_to_pack[chunk_sigs[j]]
                    drops_cache[key] = pack

            finally:
                if old_tc is not None:
                    setattr(model, "target_class", old_tc)
                if old_mask is not None:
                    setattr(model, "mask", old_mask)
                del Xb, Mb
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            chunk_counter += 1
            print(f"... chunk={len(chunk)} (cum {start + len(chunk)}/{len(tasks)})")

    return drops_cache
