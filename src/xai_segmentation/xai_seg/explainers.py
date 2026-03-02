import gc
import time
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, NoiseTunnel, Saliency
from joblib import Parallel, delayed
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .xai_helpers import get_repeated_mask, mask_and_reduce
from .xai_metrics import DropsPackFaith, calculate_all_metrics

Tensor = torch.Tensor
Metrics = Dict[str, Union[float, Tensor]]


def _cleanup(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync_cuda(x: Tensor):
    if isinstance(x, torch.Tensor) and x.is_cuda:
        torch.cuda.synchronize()


def _ig_baselines(X: Tensor, baseline: str) -> Tensor:
    if baseline == "0":
        return torch.zeros_like(X)
    if baseline == "mean":
        return X.mean(dim=(2, 3), keepdim=True).expand_as(X)
    raise ValueError(f"Unknown IG baseline: {baseline!r}")


class ExplainerBase:
    def __init__(self):
        self._metrics_extra: Dict = {}

    def attribute(
        self,
        X: Tensor,
        model: nn.Module,
        *,
        calculate_metrics: bool = True,
        drops: Optional[DropsPackFaith] = None,
    ) -> Union[Tensor, Tuple[Tensor, Metrics, float]]:
        if calculate_metrics:
            _sync_cuda(X)
            t0 = time.perf_counter()

        attr_full = self._compute(X, model)
        attr = attr_full if attr_full.ndim == 2 else mask_and_reduce(attr_full, model)

        if not calculate_metrics:
            return attr

        _sync_cuda(X)
        time_attr_s = time.perf_counter() - t0
        metrics = calculate_all_metrics(
            self._metrics_callable(),
            attr,
            X,
            model,
            drops,
            **self._metrics_kwargs(),
        )
        return attr, metrics, time_attr_s

    def _compute(self, X: Tensor, model: nn.Module) -> Tensor:
        s = Saliency(model)
        try:
            return s.attribute(X, abs=True)
        finally:
            _cleanup(s)

    def _metrics_callable(self) -> Callable:
        def fn(Xi, mi, **_):
            s = Saliency(mi)
            try:
                out = s.attribute(Xi, abs=True)
            finally:
                _cleanup(s)
            return out if out.ndim == 2 else mask_and_reduce(out, mi)

        return fn

    def _metrics_kwargs(self) -> Dict:
        return self._metrics_extra


class SaliencyExplainer(ExplainerBase):
    def __init__(self, use_absolute: bool = True):
        super().__init__()
        self.use_absolute = use_absolute
        self._metrics_extra = {"use_absolute": use_absolute}

    def _compute(self, X, model):
        s = Saliency(model)
        try:
            return s.attribute(X, abs=self.use_absolute)
        finally:
            _cleanup(s)

    def _metrics_callable(self):
        def fn(Xi, mi, **_):
            s = Saliency(mi)
            try:
                out = s.attribute(Xi, abs=self.use_absolute)
            finally:
                _cleanup(s)
            return out if out.ndim == 2 else mask_and_reduce(out, mi)

        return fn


class SmoothGradExplainer(ExplainerBase):
    def __init__(
        self,
        stdevs: float = 0.02,
        nt_samples: int = 16,
        nt_samples_batch_size: int = 16,
        use_absolute: bool = True,
    ):
        super().__init__()
        self.stdevs = stdevs
        self.nt_samples = nt_samples
        self.nt_samples_batch_size = nt_samples_batch_size
        self.use_absolute = use_absolute
        self._metrics_extra = dict(
            stdevs=stdevs,
            nt_samples=nt_samples,
            nt_samples_batch_size=nt_samples_batch_size,
            use_absolute=use_absolute,
        )

    def _compute(self, X, model):
        sal = Saliency(model)
        nt = NoiseTunnel(sal)
        try:
            return nt.attribute(
                X,
                nt_type="smoothgrad",
                stdevs=self.stdevs,
                nt_samples=self.nt_samples,
                nt_samples_batch_size=self.nt_samples_batch_size,
                abs=self.use_absolute,
            )
        finally:
            _cleanup(nt, sal)

    def _metrics_callable(self):
        def fn(Xi, mi, **_):
            sal = Saliency(mi)
            nt = NoiseTunnel(sal)
            try:
                out = nt.attribute(
                    Xi,
                    nt_type="smoothgrad",
                    stdevs=self.stdevs,
                    nt_samples=self.nt_samples,
                    nt_samples_batch_size=self.nt_samples_batch_size,
                    abs=self.use_absolute,
                )
            finally:
                _cleanup(nt, sal)
            return out if out.ndim == 2 else mask_and_reduce(out, mi)

        return fn


class IGExplainer(ExplainerBase):
    def __init__(self, baseline: str = "0", n_steps: int = 10, internal_batch_size: int = 10):
        super().__init__()
        self.baseline = baseline
        self.n_steps = n_steps
        self.internal_batch_size = internal_batch_size
        self._metrics_extra = dict(baseline=baseline, n_steps=n_steps, internal_batch_size=internal_batch_size)

    def _compute(self, X, model):
        ig = IntegratedGradients(model)
        bl = _ig_baselines(X, self.baseline)
        try:
            return ig.attribute(X, baselines=bl, n_steps=self.n_steps, internal_batch_size=self.internal_batch_size)
        finally:
            _cleanup(ig)

    def _metrics_callable(self):
        def fn(Xi, mi, **_):
            ig = IntegratedGradients(mi)
            bl = _ig_baselines(Xi, self.baseline)
            try:
                out = ig.attribute(
                    Xi,
                    baselines=bl,
                    n_steps=self.n_steps,
                    internal_batch_size=self.internal_batch_size,
                )
            finally:
                _cleanup(ig)
            return out if out.ndim == 2 else mask_and_reduce(out, mi)

        return fn


class RegionOcclusionExplainer(ExplainerBase):
    def __init__(self, baseline: str = "0", sigma_noise: float = 0.4, channel_chunk: int = 13, n_noise: int = 3):
        super().__init__()
        self.baseline = baseline
        self.sigma_noise = sigma_noise
        self.channel_chunk = channel_chunk
        self.n_noise = n_noise
        self._metrics_extra = dict(baseline=baseline, sigma_noise=sigma_noise)

    def _compute(self, X, model):
        B, C, H, W = X.shape
        dev, dt = X.device, X.dtype

        m_spatial = model._mask_broadcast(B, dtype=dt, device=dev).bool()
        m_sp = m_spatial.squeeze(1) if (m_spatial.ndim == 4 and m_spatial.size(1) == 1) else m_spatial

        if self.baseline == "mean":
            ch_mean = X.mean(dim=(2, 3), keepdim=True)
        if self.baseline == "0":
            ch_min = X.amin(dim=(2, 3), keepdim=True)

        with torch.inference_mode():
            base = model(X).view(B)

        attr = torch.zeros(B, C, device=dev, dtype=torch.float32)

        for start in range(0, C, self.channel_chunk):
            chs = list(range(start, min(start + self.channel_chunk, C)))
            k = len(chs)
            reps = self.n_noise if (self.baseline == "noise" and self.n_noise > 1) else 1
            y_sum = torch.zeros(B, k, device=dev, dtype=torch.float32)

            for _ in range(reps):
                Xb = X.repeat_interleave(k, dim=0)
                Xb_view = Xb.view(B, k, C, H, W)

                for j, ch in enumerate(chs):
                    vals = Xb_view[:, j, ch, :, :]
                    if self.baseline == "0":
                        mu = ch_min[:, ch, 0, 0].view(B, 1, 1)
                        vals[m_sp] = mu.expand_as(vals)[m_sp]
                    elif self.baseline == "mean":
                        mu = ch_mean[:, ch, 0, 0].view(B, 1, 1)
                        vals[m_sp] = mu.expand_as(vals)[m_sp]
                    else:
                        std_ch = X[:, ch, :, :].reshape(B, -1).std(dim=1).view(B, 1, 1)
                        eps = torch.randn_like(vals)
                        vals_pert = (vals + self.sigma_noise * std_ch * eps).clamp(0, 1)
                        vals[m_sp] = vals_pert[m_sp]

                with torch.inference_mode():
                    yb = model(Xb).view(B, k)
                y_sum += yb.float()

                del Xb, Xb_view, yb
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            y_mean = y_sum / float(reps)
            attr[:, start : start + k] = base.view(B, 1) - y_mean

        if self.baseline == "mean":
            del ch_mean
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return attr


class LIMEExplainer(ExplainerBase):
    def __init__(
        self,
        baseline: str = "noise",
        sigma_noise: float = 0.0,
        n_samples: int = 150,
        p: float = 0.5,
        batch_size: int = 64,
    ):
        super().__init__()
        self.baseline = baseline
        self.sigma_noise = sigma_noise
        self.n_samples = n_samples
        self.p = p
        self.batch_size = batch_size
        self._metrics_extra = dict(
            baseline=baseline,
            n_samples=n_samples,
            p=p,
            lime_batch_size=batch_size,
            **({"sigma_noise": sigma_noise} if baseline == "noise" else {}),
        )

    def _compute(self, X, model):
        B, C, H, W = X.shape
        dev, dt = X.device, X.dtype

        m = getattr(model, "mask", None)
        if m is None:
            mask_sum_vec = torch.full((B,), float(H * W), device=dev, dtype=torch.float32)
        elif m.ndim == 2:
            mask_sum_vec = torch.full((B,), float(m.sum().item()), device=dev, dtype=torch.float32)
        else:
            mask_sum_vec = m.view(B, -1).sum(1).to(device=dev, dtype=torch.float32)

        band_std = X.view(B, C, -1).std(dim=2)
        if not (self.baseline == "noise" and self.sigma_noise != 0.0):
            band_std = None

        with torch.inference_mode():
            score_o = model(X, not_agg=True)

        diffs_lists = [[] for _ in range(B)]
        bm_lists = [[] for _ in range(B)]
        done = 0

        if self.baseline == "mean":
            ch_bg_mean = X.mean(dim=(2, 3), keepdim=True)
        else:
            ch_bg_mean = None

        if self.baseline == "0":
            ch_min = X.amin(dim=(2, 3), keepdim=True)
        else:
            ch_min = None

        rep_mask_base = get_repeated_mask(model, B=B, channels=C).to(device=dev, dtype=dt)

        while done < self.n_samples:
            Bp = min(self.batch_size, self.n_samples - done)

            X_rep = X.repeat_interleave(Bp, dim=0)

            if band_std is not None:
                std_rep = band_std.repeat_interleave(Bp, dim=0)
            else:
                std_rep = None

            gate = torch.empty((B, Bp, C), device=dev).bernoulli_(self.p).to(dt)
            gate_flat = gate.reshape(B * Bp, C)

            eff_flat = gate_flat if self.baseline == "noise" else (1.0 - gate_flat)
            bm_eff = eff_flat.view(B, Bp, C)

            rep_mask = rep_mask_base.repeat_interleave(Bp, dim=0)
            eff = eff_flat.view(B * Bp, C, 1, 1)
            mg_sp = eff * rep_mask

            if self.baseline == "noise":
                noise = torch.randn_like(X_rep)
                if std_rep is not None:
                    std_map = std_rep.view(B * Bp, C, 1, 1).to(X_rep.device, X_rep.dtype)
                    Xb = (X_rep + self.sigma_noise * noise * std_map * mg_sp).clamp(0, 1)
                else:
                    Xb = (X_rep + self.sigma_noise * noise * mg_sp * X_rep).clamp(0, 1)
            elif self.baseline == "0":
                min_rep = ch_min.repeat_interleave(Bp, dim=0)
                Xb = (X_rep * (1.0 - mg_sp) + min_rep * mg_sp).clamp(0, 1)
            else:
                mean_rep = ch_bg_mean.repeat_interleave(Bp, dim=0)
                Xb = (X_rep * (1.0 - mg_sp) + mean_rep * mg_sp).clamp(0, 1)

            with torch.inference_mode():
                score_p_flat = model(Xb, not_agg=True)

            score_p = score_p_flat.view(B, Bp, H, W)
            so = score_o.unsqueeze(1).expand(B, Bp, H, W)
            diff = (so - score_p).view(B, Bp, -1).sum(dim=2) / mask_sum_vec.view(B, 1)

            diff_cpu = diff.detach().cpu().numpy()
            bm_cpu = bm_eff.detach().cpu().numpy()
            for i in range(B):
                diffs_lists[i].append(diff_cpu[i])
                bm_lists[i].append(bm_cpu[i])

            done += Bp

            del X_rep, Xb, rep_mask, gate, gate_flat, bm_eff, score_p_flat, score_p, so, diff, mg_sp, eff, eff_flat
            if std_rep is not None:
                del std_rep
            if self.baseline == "noise":
                del noise
                if "std_map" in locals():
                    del std_map
            if self.baseline == "0":
                del min_rep
            if self.baseline == "mean":
                del mean_rep

        alphas = np.logspace(-7, 0, 60)
        n_folds = 5 if self.n_samples >= 50 else 3
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        def _fit_one(bm_list, df_list):
            bm = np.concatenate(bm_list, axis=0)
            df = np.concatenate(df_list, axis=0)

            if self.baseline in ("0", "mean"):
                C_local = bm.shape[1]
                kernel_width = 0.75 * np.sqrt(C_local)
                dist = np.sqrt((bm.astype(np.float32) ** 2).sum(axis=1))
                w_full = np.exp(-(dist**2) / (kernel_width**2)).astype(np.float64)
            else:
                w_full = np.ones_like(df, dtype=np.float64)

            uniq_X, inv = np.unique(bm, axis=0, return_inverse=True)
            y = np.zeros(uniq_X.shape[0], dtype=np.float64)
            w = np.zeros(uniq_X.shape[0], dtype=np.float64)
            np.add.at(y, inv, df.astype(np.float64) * w_full)
            np.add.at(w, inv, w_full)
            w = np.maximum(w, 1e-12)
            y /= w
            sample_weight = w

            Xs = StandardScaler().fit_transform(uniq_X)
            lcv = LassoCV(
                alphas=alphas,
                cv=cv,
                max_iter=2500,
                random_state=42,
                fit_intercept=True,
                precompute=False,
                n_jobs=1,
            )
            lcv.fit(Xs, y, sample_weight=sample_weight)

            m = lcv.mse_path_.mean(axis=1)
            s = lcv.mse_path_.std(axis=1) / np.sqrt(lcv.mse_path_.shape[1])
            i_min = np.argmin(m)
            thr = m[i_min] + s[i_min]
            idx = np.where(m <= thr)[0]
            alpha_star = lcv.alphas_[idx[-1] if len(idx) else i_min]

            l = Lasso(
                alpha=alpha_star,
                max_iter=10000,
                random_state=42,
                fit_intercept=True,
                precompute=False,
            )
            l.fit(Xs, y, sample_weight=sample_weight)
            return l.coef_.astype(np.float32)

        coefs = Parallel(n_jobs=-1, backend="loky")(
            delayed(_fit_one)(bm_lists[i], diffs_lists[i]) for i in range(B)
        )
        attr = torch.stack([torch.from_numpy(c).to(dev) for c in coefs], dim=0)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return attr

    def _metrics_callable(self):
        def fn(Xi, mi, **_):
            return self._compute(Xi, mi)

        return fn
