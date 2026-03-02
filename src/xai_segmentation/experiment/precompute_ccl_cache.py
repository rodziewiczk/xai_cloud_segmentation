import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from skimage.measure import label as sk_label

from experiment.dataclass import CCLCache, CCLItem
from padding.padding import maybe_pad_to_size

DropsCache = Dict[Tuple[str, Any, Any, int, Optional[int], int, int, str], "DropsPackFaith"]  # tylko dla spójności typów


def precompute_ccl_cache(
        model: nn.Module,
        dataset,
        classes: List[int],
        *,
        normalize_fn,
        connectivity: int = 4,
        B_pred: int = 100,
        device: Optional[torch.device] = None,
        ccl_workers: Optional[int] = None,
        pad_to_hw: Optional[tuple[int, int]] = None,
        pad_mode: str = "constant",
        pad_value: float = 0.0,
        persist_ccl_dir: Optional[str] = None,
) -> Tuple["CCLCache", pd.DataFrame, Dict[Any, Tuple[torch.Tensor, dict]]]:
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    sk_conn = 1 if connectivity == 4 else 2
    ccl_workers = ccl_workers or min(8, os.cpu_count() or 4)
    if persist_ccl_dir: os.makedirs(persist_ccl_dir, exist_ok=True)

    def _save_ccl(path_str, image_id, class_id, labels, sizes, H, W, pad_info):
        if not persist_ccl_dir: return
        base_noext = os.path.splitext(os.path.basename(os.path.abspath(path_str)))[0] or "sample"
        out_path = os.path.join(persist_ccl_dir, f"{base_noext}_img{int(image_id)}_cls{int(class_id)}.ccl.npz")
        np.savez_compressed(
            out_path,
            labels=labels.astype(np.int16, copy=False),
            sizes=sizes.astype(np.int32, copy=False),
            H=np.int32(H), W=np.int32(W),
            class_id=np.int16(class_id), image_id=np.int32(image_id),
            path=np.array(os.path.abspath(path_str)),
            connectivity=np.int16(connectivity),
            pad_info=np.array(json.dumps(pad_info or {})),
        )

    ccl_cache: CCLCache = {}
    rows: List[dict] = []
    image_bank: List[Tuple[Any, Tuple[torch.Tensor, dict]]] = []
    batch_x: List[torch.Tensor] = []
    batch_meta: List[dict] = []

    def _predict_argmax_maps(xs: List[torch.Tensor]) -> List[torch.Tensor]:
        if not xs: return []
        with torch.inference_mode():
            X = torch.stack(xs, 0).to(device)
            logits = model._get_logits(X) if hasattr(model, "_get_logits") else model(X)
            return [p.detach().cpu() for p in logits.argmax(1)]

    def _label_one(mask_bool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lab32 = sk_label(mask_bool.astype(np.uint8, copy=False), connectivity=sk_conn)
        sizes = np.bincount(lab32.ravel()).astype(np.int32, copy=False)
        return lab32.astype(np.int16, copy=False), sizes

    def _flush_batch():
        nonlocal batch_x, batch_meta
        if not batch_x: return
        pred_maps = _predict_argmax_maps(batch_x)

        tasks: List[Tuple[str, Any, int, str, int, int, np.ndarray, Optional[dict]]] = []
        for (x_norm, meta), pred in zip(zip(batch_x, batch_meta), pred_maps):
            dname = meta.get("dataset_name", "dataset")
            iid = meta.get("image_id")
            path = meta.get("path", "")
            pad_i = meta.get("pad_info")
            H, W = int(x_norm.shape[-2]), int(x_norm.shape[-1])
            image_bank.append((iid, (x_norm.detach().cpu(), meta)))
            for clazz in classes:
                m = (pred == int(clazz)).numpy()
                if m.any():
                    tasks.append((dname, iid, int(clazz), path, H, W, m, pad_i))

        results: List[Tuple[str, Any, int, str, int, int, np.ndarray, np.ndarray, Optional[dict]]] = []
        if ccl_workers == 1 or len(tasks) <= 1:
            for dname, iid, cid, path, H, W, m, pad_i in tasks:
                lab, sizes = _label_one(m)
                results.append((dname, iid, cid, path, H, W, lab, sizes, pad_i))
        else:
            with ThreadPoolExecutor(max_workers=int(ccl_workers)) as ex:
                futs = [(dname, iid, cid, path, H, W, pad_i, ex.submit(_label_one, m)) for
                        dname, iid, cid, path, H, W, m, pad_i in tasks]
                for dname, iid, cid, path, H, W, pad_i, fut in futs:
                    lab, sizes = fut.result()
                    results.append((dname, iid, cid, path, H, W, lab, sizes, pad_i))

        for dname, iid, cid, path, H, W, labels, sizes, pad_i in results:
            ccl_cache[(dname, iid, cid)] = CCLItem(labels=labels, sizes=sizes, H=H, W=W, pad_info=pad_i)
            _save_ccl(path, iid, cid, labels, sizes, H, W, pad_i)
            for comp_idx in range(1, int(sizes.size)):
                rows.append(dict(
                    dataset_name=dname, image_id=iid, path=path,
                    class_id=cid, component_idx=int(comp_idx),
                    pixel_count=int(sizes[comp_idx]), connectivity=int(connectivity),
                    H=H, W=W, pad_info=pad_i, cache_key=(dname, iid, cid),
                ))

        batch_x, batch_meta = [], []

    for x_raw, meta in dataset:
        x = x_raw if isinstance(x_raw, torch.Tensor) else torch.from_numpy(x_raw)
        if x.ndim == 2: x = x.unsqueeze(0)
        x = normalize_fn(x.float())
        x, pad_info = maybe_pad_to_size(x, target_hw=pad_to_hw, mode=pad_mode, value=pad_value)
        m = dict(meta);
        m["pad_info"] = pad_info
        batch_x.append(x);
        batch_meta.append(m)
        if len(batch_x) >= B_pred: _flush_batch()
    _flush_batch()

    image_bank_dict: Dict[Any, Tuple[torch.Tensor, dict]] = {iid: pack for iid, pack in image_bank}
    index_df = pd.DataFrame(rows)
    return ccl_cache, index_df, image_bank_dict
