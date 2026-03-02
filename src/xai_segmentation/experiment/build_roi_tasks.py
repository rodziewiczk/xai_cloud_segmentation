from typing import List, Dict, Tuple, Any

import pandas as pd

from experiment.dataclass import CCLCache


def build_roi_tasks(
        ccl_cache: CCLCache,
        index_df: pd.DataFrame,
        *,
        pixel_count: int,
) -> Tuple[List[dict], List[dict]]:
    locals_tasks: List[dict] = []
    globals_tasks: List[dict] = []

    if index_df.empty:
        return locals_tasks, globals_tasks

    path_map: Dict[Tuple[Any, Any], str] = {}
    for _, r in index_df.iterrows():
        path_map[(r["dataset_name"], r["image_id"])] = r.get("path", "")

    grouped = index_df.groupby(["dataset_name", "image_id", "class_id"], dropna=False)
    for (dname, iid, cid), grp in grouped:
        key = (dname, iid, int(cid))
        ccl = ccl_cache.get(key)
        if ccl is None:
            continue

        n_comp = int(ccl.sizes.size - 1)
        union_area = int(int(ccl.sizes.sum()) - int(ccl.sizes[0]))
        has_global = (union_area >= int(pixel_count))

        local_rows = grp[grp["pixel_count"] >= int(pixel_count)]
        has_locals = not local_rows.empty

        if has_global:
            comp_tuple = tuple(range(1, n_comp + 1))  
            mask_sig = f"{dname}|{iid}|{int(cid)}|{','.join(map(str, comp_tuple))}"
            globals_tasks.append(dict(
                roi_type="global", dataset_name=dname, image_id=iid, path=path_map.get((dname, iid), ""),
                class_id=int(cid), component_idx=None, pixel_count=union_area, H=int(ccl.H), W=int(ccl.W),
                mask_sig=mask_sig,
            ))

        if has_locals:
            for _, rr in local_rows.iterrows():
                comp_idx = int(rr["component_idx"])
                comp_tuple = (comp_idx,)
                mask_sig = f"{dname}|{iid}|{int(cid)}|{','.join(map(str, comp_tuple))}"
                locals_tasks.append(dict(
                    roi_type="local", dataset_name=dname, image_id=iid, path=rr.get("path", ""),
                    class_id=int(cid), component_idx=comp_idx, pixel_count=int(rr["pixel_count"]),
                    H=int(ccl.H), W=int(ccl.W),
                    mask_sig=mask_sig,
                ))

    return locals_tasks, globals_tasks
