import torch


def get_attributions(explainer, model, X: torch.Tensor, **kwargs) -> torch.Tensor:
    if callable(explainer):
        attr = explainer(X, model, calculate_metrics=False, **kwargs)
    else:
        attr = explainer.attribute(X, **kwargs)

    if attr.ndim == 4:
        attr = attr.sum(dim=(2, 3))

    return attr


def get_repeated_mask(model, B: int = 1, channels: int = 13) -> torch.Tensor:
    mask = model.mask

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)

    if mask.size(0) == 1:
        mask = mask.expand(B, -1, -1)  # (B, H, W)

    return mask.unsqueeze(1).expand(B, channels, mask.size(1), mask.size(2))


def mask_and_reduce(attr_full: torch.Tensor, model) -> torch.Tensor:
    if attr_full.ndim == 4:
        B, C, H, W = attr_full.shape
        rep_mask = get_repeated_mask(model, B=B, channels=C)  # (B,C,H,W)
        return (attr_full * rep_mask).sum(dim=(2, 3))  # (B,C)
    elif attr_full.ndim == 3:
        C, H, W = attr_full.shape
        rep_mask = get_repeated_mask(model, B=1, channels=C)  # (1,C,H,W)
        return (attr_full.unsqueeze(0) * rep_mask).sum(dim=(2, 3)).squeeze(0)  # (C,)
    else:
        raise ValueError(f"mask_and_reduce expects (B,C,H,W) or (C,H,W), got {tuple(attr_full.shape)}")
