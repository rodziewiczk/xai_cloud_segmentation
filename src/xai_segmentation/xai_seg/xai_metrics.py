from dataclasses import dataclass
import torch
import gc
from .xai_helpers import get_attributions


@dataclass
class DropsPackFaith:
    drops_all_raw: torch.Tensor
    b_count_matrix: torch.Tensor



# po eksperymentach zmieniono definicje metryki sparsity, bo ta nowa jest bardziej elegancka, nie zmieniło to żadnych wniosków
def sparsity(original_attributions: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    x = original_attributions
    if x.ndim == 1:
        x = x.unsqueeze(0)
    abs_vals = x.abs()
    sums = abs_vals.sum(dim=1, keepdim=True).clamp_min(1e-12)
    norm_vals = abs_vals / sums
    sparsity = (norm_vals > threshold).float().mean(dim=1)
    return sparsity


def faithfulness(original_attributions: torch.Tensor, drops) -> torch.Tensor:
    dar = drops.drops_all_raw
    bcm = drops.b_count_matrix
    attr = original_attributions
    sums_attr_all_raw = (bcm * attr.unsqueeze(1)).sum(dim=-1)

    def pearson_rowwise(x: torch.Tensor, y: torch.Tensor, correction: int = 1) -> torch.Tensor:
        x = x.to(torch.float64)
        y = y.to(torch.float64)
        n = x.size(1)
        xm = x.mean(dim=1, keepdim=True)
        ym = y.mean(dim=1, keepdim=True)
        xc = x - xm
        yc = y - ym
        denom_n = max(1, n - correction)
        cov = (xc * yc).sum(dim=1) / denom_n
        var_x = (xc * xc).sum(dim=1) / denom_n
        var_y = (yc * yc).sum(dim=1) / denom_n
        denom = (var_x.clamp_min(0) * var_y.clamp_min(0)).sqrt()
        r = torch.where(denom == 0, torch.zeros_like(denom), cov / denom)
        return r.to(torch.float32)

    f_raw = pearson_rowwise(sums_attr_all_raw, dar)
    return f_raw


def robustness(
        explainer,
        original_attributions: torch.Tensor,
        model,
        X: torch.Tensor,
        n_robustness: int = 3,
        sigma_robustness: float = 0.1,
        robustness_batch_size: int = 1,
        **kwargs
):
    device = X.device
    B, C, H, W = X.shape
    eps = 1e-12
    orig = original_attributions
    orig_unit = orig / (orig.norm(dim=1, keepdim=True) + eps)  # (B,C)
    dists_per_image = [[] for _ in range(B)]

    t = 0
    while t < n_robustness:
        Bp = min(robustness_batch_size, n_robustness - t)
        X_rep = X.unsqueeze(1).expand(B, Bp, C, H, W).contiguous().view(B * Bp, C, H, W)
        m_sp = model._mask_broadcast(B * Bp, dtype=X.dtype, device=device)
        m_sp = m_sp.unsqueeze(1).expand(B * Bp, C, H, W)
        noise = torch.randn_like(X_rep)
        Xb = (X_rep + sigma_robustness * m_sp * noise * X_rep).clamp_(0, 1)

        del X_rep, m_sp, noise

        attr_flat = get_attributions(explainer, model, Xb, **kwargs)
        attr_bb = attr_flat.view(B, Bp, C)
        pert_unit = attr_bb / (attr_bb.norm(dim=2, keepdim=True) + eps)
        d = (orig_unit.unsqueeze(1) - pert_unit).norm(dim=2)

        for i in range(B):
            dists_per_image[i].append(d[i])

        del Xb, attr_flat, attr_bb, pert_unit, d

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        t += Bp

    dist_means = torch.stack([torch.cat(chunks).mean() for chunks in dists_per_image], dim=0)
    return dist_means


def calculate_drops_batch(
    model,
    X: torch.Tensor,
    frac: float = 0.125,
    n_faith: int = 500,
    faith_batch_size: int = 100,
):
    
    
    device = X.device
    B, C, H, W = X.shape

    with torch.inference_mode():
        base = model(X)
        base = base.view(B, -1).mean(dim=1) if base.ndim > 1 else base.view(B)

    m = model._mask_broadcast(B, dtype=X.dtype, device=device).bool()

    idx2d_list, M_list = [], []
    for i in range(B):
        idx2d = torch.nonzero(m[i], as_tuple=False)
        K_i = idx2d.size(0)
        m_i = min(K_i, max(1, int(frac * max(1, K_i) * C)))
        if K_i == 0 or m_i == 0:
            hi = wi = torch.empty(0, dtype=torch.long, device=device)
        else:
            perm = torch.randperm(K_i, device=device)[:m_i]
            hi, wi = idx2d[perm].t()
        idx2d_list.append((hi, wi))
        M_list.append(hi.numel())

    drops_all_raw = torch.zeros(B, n_faith, device=device, dtype=torch.float32)
    b_count_matrix = torch.zeros(B, n_faith, C, device=device, dtype=torch.float32)
    dirichlet = torch.distributions.Dirichlet(torch.ones(C, device=device))

    orig_mask = getattr(model, "mask", None)

    t0 = 0
    while t0 < n_faith:
        T_block = min(n_faith - t0, max(1, faith_batch_size))

        i_start = 0
        while i_start < B:

            max_imgs_for_this_T = max(1, faith_batch_size // T_block)
            if max_imgs_for_this_T == 0:
                T_block = max(1, faith_batch_size)
                max_imgs_for_this_T = 1

            B_chunk = min(B - i_start, max_imgs_for_this_T)
            i_end = i_start + B_chunk

            X_chunk = X[i_start:i_end]  # (B_chunk,C,H,W)
            Xb = (
                X_chunk.unsqueeze(1)
                .expand(B_chunk, T_block, C, H, W)
                .contiguous()
                .view(B_chunk * T_block, C, H, W)
            )

            for local_i, i in enumerate(range(i_start, i_end)):
                hi, wi = idx2d_list[i]
                M_i = M_list[i]
                if M_i == 0:
                    continue

                P = dirichlet.sample((T_block,))  # (T_block,C)
                b = torch.multinomial(P, num_samples=M_i, replacement=True)  # (T_block,M_i)

                counts = torch.nn.functional.one_hot(b, num_classes=C).sum(dim=1)  # (T_block,C)
                b_count_matrix[i, t0 : t0 + T_block] = counts.to(b_count_matrix.dtype)

                row_base = local_i * T_block
                for ch in range(C):
                    mask_t = (b == ch)         # (T_block,M_i)
                    ks = mask_t.sum(dim=1)     # (T_block,)
                    todo = torch.nonzero(ks >= 2, as_tuple=False).squeeze(1)
                    if todo.numel() == 0:
                        continue
                    for t in todo.tolist():
                        sel = mask_t[t]
                        h_sel = hi[sel]
                        w_sel = wi[sel]
                        if h_sel.numel() < 2:
                            continue
                        r = row_base + int(t)
                        vals = Xb[r, ch, h_sel, w_sel]
                        perm_idx = torch.randperm(h_sel.numel(), device=device)
                        Xb[r, ch, h_sel, w_sel] = vals[perm_idx]

            model.mask = orig_mask
            if isinstance(orig_mask, torch.Tensor) and orig_mask.dim() == 3 and orig_mask.size(0) == B:
                model.mask = orig_mask[i_start:i_end]

        
            with torch.inference_mode():
                yb = model(Xb)
                if yb.ndim > 2:
                    yb = yb.view(B_chunk, T_block, -1).mean(dim=2)
                elif yb.ndim == 1:
                    yb = yb.view(B_chunk, T_block)
                else:
                    yb = yb.view(B_chunk, T_block)

            base_blk = base[i_start:i_end].view(B_chunk, 1).expand(-1, T_block)
            drops_all_raw[i_start:i_end, t0 : t0 + T_block] = (base_blk - yb)

            del Xb, yb, X_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            i_start = i_end

        t0 += T_block

    model.mask = orig_mask

    del m, idx2d_list, M_list, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return drops_all_raw, b_count_matrix



def calculate_all_metrics(
        explainer,
        original_attributions: torch.Tensor,
        X: torch.Tensor,
        model,
        drops: DropsPackFaith = None,
        sparsity_threshold: float = 1/13,
        n_robustness: int = 3,
        sigma_robustness: float = 0.1,
        robustness_batch_size=1,
        **kwargs
) -> torch.Tensor:
    device = X.device
    s = sparsity(original_attributions, sparsity_threshold)
    r = robustness(
        explainer, original_attributions, model, X,
        n_robustness=n_robustness, sigma_robustness=sigma_robustness,
        robustness_batch_size=robustness_batch_size, **kwargs
    )
    f_raw = faithfulness(original_attributions, drops)
    out = torch.stack([s.to(device), r.to(device), f_raw.to(device)], dim=1)

    return out
