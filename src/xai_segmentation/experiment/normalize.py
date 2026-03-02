import torch


def ftw_normalize(x: torch.Tensor) -> torch.Tensor:
    return (x / 3000.0).clamp(0, 1)

def cloudsen_normalize(x: torch.Tensor) -> torch.Tensor:
    return x

def identity_normalize(x: torch.Tensor) -> torch.Tensor:
    return x
