import torch
import torch.nn as nn
import torch
import torch.nn.functional as F


class AggSegModel(nn.Module):
    def __init__(self, base_model: nn.Module, target_class: int, mask: torch.Tensor) -> None:
        super().__init__()
        self.base = base_model
        self.base.eval()
        self.target_class = int(target_class)
        self.mask = mask

    def _get_logits(self, X: torch.Tensor) -> torch.Tensor:
        # unify to (B,C,H,W)
        if self.base.name == "UnetMobV2_V1":
            out = self.base.model(X)
        elif self.base.name == "u-efficientnet-b3":
            out = self.base(X)
        else:
            raise ValueError("Unsupported base model")

        if isinstance(out, (list, tuple)):
            out = out[0]
        if out.dim() == 3:
            out = out.unsqueeze(0)
        if out.dim() != 4:
            raise ValueError(f"Got {tuple(out.shape)}")
        return out

    def _mask_broadcast(
            self,
            B: int,
            dtype: torch.dtype,
            device: torch.device,
    ) -> torch.Tensor:
        # broadcast mask to (B,H,W)
        m = self.mask
        if m.device != device:
            m = m.to(device)
        if m.dim() == 2:
            m = m.unsqueeze(0).expand(B, -1, -1)
        elif m.dim() == 3:
            Bm = m.size(0)
            if Bm == 1:
                m = m.expand(B, -1, -1)
            elif Bm == B:
                pass
            elif B % Bm == 0:
                m = m.repeat_interleave(B // Bm, dim=0)
        return m.to(dtype=dtype)


    def forward(self, X: torch.Tensor, not_agg: bool = False, probs: bool = False) -> torch.Tensor:
       
        logits = self._get_logits(X)                # (B,C,H,W)
        B, C, H, W = logits.shape
        m = self._mask_broadcast(B, logits.dtype, logits.device)  # (B,H,W)

        if probs:
            score_map = F.softmax(logits, dim=1)[:, self.target_class, :, :]  # (B,H,W)
        else:
            score_map = logits[:, self.target_class, :, :]                  # (B,H,W)

        out_spatial = score_map * m                                         # (B,H,W)
        return out_spatial if not_agg else out_spatial.sum(dim=(1, 2))      # (B,)

