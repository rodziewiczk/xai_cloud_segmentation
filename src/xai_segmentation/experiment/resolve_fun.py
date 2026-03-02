from xai_seg.explainers import (
    SaliencyExplainer,
    SmoothGradExplainer,
    IGExplainer,
    RegionOcclusionExplainer,
    LIMEExplainer,
)

def _pick(name: str):
    n = (name or "").lower()
    if n in {"vanilla_grad", "saliency", "vanilla"}:
        return SaliencyExplainer, {"use_absolute"}, {}
    if "smoothgrad" in n or "smooth_grad" in n:
        return SmoothGradExplainer, {"stdevs", "nt_samples", "nt_samples_batch_size", "use_absolute"}, {}
    if n.startswith("ig"):
        return IGExplainer, {"baseline", "n_steps", "internal_batch_size"}, {}
    if n.startswith("reg_occ") or n == "region_occlusion" or n.startswith("occlusion"):
        return RegionOcclusionExplainer, {"baseline", "sigma_noise", "channel_chunk", "n_noise"}, {}
    if n.startswith("lime"):
        return LIMEExplainer, {"baseline", "sigma_noise", "n_samples", "p", "batch_size"}, {"lime_batch_size": "batch_size"}
    return SaliencyExplainer, {"use_absolute"}, {}

def resolve_explainer_and_kwargs(method_name: str, cfg: dict):
    cfg = dict(cfg or {})
    cfg.pop("B_method", None)
    Cls, allowed, rename = _pick(method_name)
    ctor_kwargs = {}
    for k, v in cfg.items():
        kk = rename.get(k, k)
        if kk in allowed:
            ctor_kwargs[kk] = v
    base_kwargs = dict(cfg)
    explainer = Cls(**ctor_kwargs)

    def fn(X, model, *, calculate_metrics=True, drops=None, **extra):
        return explainer.attribute(X, model, calculate_metrics=calculate_metrics, drops=drops)

    return fn, base_kwargs
