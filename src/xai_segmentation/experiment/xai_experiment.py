import gc
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Iterable, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from xai_seg.agg_seg_model import AggSegModel
from xai_seg.xai_metrics import DropsPackFaith
from .build_roi_tasks import build_roi_tasks
from .helpers import drops_cfg_key, stack_packs
from .normalize import identity_normalize
from .precompute_all_drops_to_ram import precompute_all_drops_to_ram
from .precompute_ccl_cache import precompute_ccl_cache, DropsCache
from .resolve_fun import resolve_explainer_and_kwargs
from .stack_batch_for_tasks import stack_batch_for_tasks

PARAM_COLS = [
    "baseline", "stdevs", "nt_samples", "nt_samples_batch_size", "use_absolute",
    "n_steps", "internal_batch_size",
    "sigma_noise", "sigma", "p", "lime_batch_size", "perturbations_per_eval", "channel_chunk",
    "layer"
]

def _metrics_to_numpy(m):
    if torch.is_tensor(m):
        return m.detach().cpu().numpy()
    if isinstance(m, dict):
        keys = ("sparsity", "robustness", "faithfulness_r_raw")
        cols = []
        for k in keys:
            v = m.get(k)
            if torch.is_tensor(v):
                cols.append(v.detach().cpu().numpy().reshape(-1))
            else:
                cols.append(np.asarray(v, dtype=float).reshape(-1))
        return np.stack(cols, axis=1)  # (B, 3)
    return np.asarray(m, dtype=float)


def run_xai_experiment_v3(
        raw_model: nn.Module,
        dataset: Iterable[Tuple[torch.Tensor, dict]],
        classes: List[int],
        methods_cfg: Dict[str, dict],
        *,
        normalize_fn: Callable[[torch.Tensor], torch.Tensor] = identity_normalize,
        pad_to_hw: Optional[tuple[int, int]] = None,
        pad_mode: str = "constant",
        pad_value: float = 0.0,
        connectivity: int = 4,
        B_pred: int = 100,
        drops_cfg: Dict[str, Any],
        drops_masks_batch_size: int = 8,
        B_method_default: int = 4,
        pixel_count: int = 128,
        global_seed: Optional[int] = 123,
        deterministic: bool = True,
        ccl_workers: int = 8,
        persist_csv_dir: Optional[str] = None,
        persist_ccl_dir: Optional[str] = None,
        calculate_metrics: bool = False,
) -> pd.DataFrame:

    if global_seed is not None:
        import random as _random
        _random.seed(global_seed)
        np.random.seed(global_seed)
        torch.manual_seed(global_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(global_seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_mask = torch.ones(1, 1, dtype=torch.float32)
    model = AggSegModel(base_model=raw_model, target_class=0, mask=dummy_mask).to(device)
    model = model.to(device).eval()

    ccl_cache, index_df, image_bank = precompute_ccl_cache(
        model=model, dataset=dataset, classes=classes,
        normalize_fn=normalize_fn, connectivity=connectivity, B_pred=B_pred, device=device,
        ccl_workers=ccl_workers, pad_to_hw=pad_to_hw, pad_mode=pad_mode, pad_value=pad_value,
        persist_ccl_dir=persist_ccl_dir,
    )
    print("CCL DONE")

    local_tasks, global_tasks = build_roi_tasks(ccl_cache, index_df, pixel_count=pixel_count)

    drops_cache = None
    if calculate_metrics:
        drops_cache: DropsCache = precompute_all_drops_to_ram(
            model=model,
            local_tasks=local_tasks,
            global_tasks=global_tasks,
            image_bank=image_bank,
            ccl_cache=ccl_cache,
            device=device,
            drops_cfg=drops_cfg,
            drops_masks_batch_size=drops_masks_batch_size,
        )
    else:
        print("[XAIv3] Skipping DROPS precompute (compute_metrics/use_drops=False)")

    rows: List[dict] = []
    method_cache: Dict[Tuple[str, str], Tuple[torch.Tensor, np.ndarray, float]] = {}

    def _apply_method_on_tasks(method_name: str, cfg: dict, tasks: List[dict]) -> int:
        if not tasks:
            return 0
        fn, base_kwargs = resolve_explainer_and_kwargs(method_name, cfg)
        B_method_this = int(cfg.get("B_method", B_method_default))

        processed_total = 0
        byc: Dict[int, List[dict]] = {}
        for t in tasks:
            byc.setdefault(int(t["class_id"]), []).append(t)

        for cid, tlist in byc.items():
            for start in range(0, len(tlist), B_method_this):
                chunk = tlist[start:start + B_method_this]
                Xb, Mb, H, W = stack_batch_for_tasks(
                    chunk, image_bank=image_bank, ccl_cache=ccl_cache, device=device
                )

                mkeys = [(method_name, t["mask_sig"]) for t in chunk]
                to_compute_idx = [i for i, k in enumerate(mkeys) if k not in method_cache]

                old_tc = getattr(model, "target_class", None)
                old_mask = getattr(model, "mask", None)
                try:
                    setattr(model, "target_class", int(cid))

                    if to_compute_idx:
                        Xb_new = Xb[to_compute_idx]
                        Mb_new = Mb[to_compute_idx]
                        setattr(model, "mask", Mb_new)
                        if calculate_metrics:
                            cfg_key_str = drops_cfg_key(drops_cfg)
                            packs_new = []
                            for idx_in_chunk in to_compute_idx:
                                t_i = chunk[idx_in_chunk]
                                key_i = (
                                    t_i["roi_type"], t_i["dataset_name"], t_i["image_id"], int(cid),
                                    (int(t_i["component_idx"]) if t_i["roi_type"] == "local" else None),
                                    int(H), int(W), cfg_key_str
                                )
                                pack_i = drops_cache.get(key_i)
                                if pack_i is None:
                                    raise RuntimeError(f"NO DROPS in cache for ROI: {key_i}")
                                packs_new.append(pack_i)

                            pack_new = stack_packs(packs_new)
                            drops = DropsPackFaith(pack_new.drops_all_raw.to(device),
                                                   pack_new.b_count_matrix.to(device))

                        if calculate_metrics:
                            attr, metrics, time_attr_s = fn(
                                Xb_new, model, calculate_metrics=True, drops=drops, **base_kwargs
                            )
                            time_ms_attr = float(time_attr_s * 1000.0)
                            metrics_np = _metrics_to_numpy(metrics)
                        else:
                            attr = fn(Xb_new, model, calculate_metrics=False, **base_kwargs)
                            time_ms_attr = float("nan")
                            metrics_np = np.full((len(to_compute_idx), 3), np.nan, dtype=float)

                        attr_cpu = attr.detach().cpu()

                        for k, idx_in_chunk in enumerate(to_compute_idx):
                            key = mkeys[idx_in_chunk]
                            method_cache[key] = (
                                attr_cpu[k].clone(),
                                metrics_np[k].copy(),
                                time_ms_attr,
                            )

                    for i, t in enumerate(chunk):
                        cached = method_cache[mkeys[i]]
                        attr_row, metrics_row, time_ms_attr = cached[:3]
                        rec = dict(
                            dataset_name=t["dataset_name"], image_id=t["image_id"], path=t.get("path", ""),
                            roi_type=t["roi_type"], class_id=int(cid),
                            component_idx=(int(t["component_idx"]) if t["roi_type"] == "local" else np.nan),
                            pixel_count=int(t["pixel_count"]), connectivity=int(connectivity),
                            H=int(H), W=int(W),
                            method=method_name, B_method_used=B_method_this,
                            time_ms_attr=time_ms_attr,
                            sparsity=float(metrics_row[0]),
                            robustness=float(metrics_row[1]),
                            faithfulness_r_raw=float(metrics_row[2]),
                        )
                        for col in PARAM_COLS:
                            if col in base_kwargs:
                                rec[col] = base_kwargs[col]
                        vec = attr_row.flatten().tolist()
                        for j in range(len(vec)):
                            rec[f"attr_c{j}"] = float(vec[j])
                        rows.append(rec)

                    processed_total += len(chunk)

                finally:
                    if old_tc is not None: setattr(model, "target_class", old_tc)
                    if old_mask is not None: setattr(model, "mask", old_mask)
                    del Xb, Mb
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    gc.collect()

        return processed_total

    for method_name, cfg in methods_cfg.items():
        _apply_method_on_tasks(method_name, cfg, global_tasks)
        _apply_method_on_tasks(method_name, cfg, local_tasks)
        print(f"[XAIv3] Calculated — {method_name}")

    df = pd.DataFrame(rows)

    if persist_csv_dir:
        os.makedirs(persist_csv_dir, exist_ok=True)
        out_csv = os.path.join(persist_csv_dir, "xai_experiment_v3.csv")
        df.to_csv(out_csv, index=False)
        with open(os.path.join(persist_csv_dir, "xai_experiment_v3_config.json"), "w") as f:
            json.dump(dict(
                classes=list(map(int, classes)), connectivity=int(connectivity), B_pred=int(B_pred),
                drops_cfg=drops_cfg, drops_masks_batch_size=int(drops_masks_batch_size),
                B_method_default=int(B_method_default), pixel_count=int(pixel_count),
                global_seed=global_seed, deterministic=deterministic,
                pad_to_hw=pad_to_hw, pad_mode=pad_mode, pad_value=pad_value,
                calculate_metrics=calculate_metrics
            ), f, indent=2)

    return df
