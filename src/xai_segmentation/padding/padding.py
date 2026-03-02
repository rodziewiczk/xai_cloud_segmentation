import torch
import torch.nn.functional as F

def maybe_pad_to_size(x, target_hw=None, mode="constant", value=0.0):
    if target_hw is None:
        return x, None

    th, tw = int(target_hw[0]), int(target_hw[1])
    _, H, W = x.shape

    pad_h = max(0, th - H)
    pad_w = max(0, tw - W)

    top = bottom = left = right = 0
    if pad_h > 0 or pad_w > 0:
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        if mode == "constant":
            x = F.pad(x, (left, right, top, bottom), mode="constant", value=value)
        else:
            x = F.pad(x, (left, right, top, bottom), mode=mode)

    _, H2, W2 = x.shape
    crop_top = crop_left = 0
    if H2 > th or W2 > tw:
        x = x[:, crop_top:crop_top + th, crop_left:crop_left + tw]

    pad_info = {
        "pad_top": int(top), "pad_bottom": int(bottom),
        "pad_left": int(left), "pad_right": int(right),
        "crop_top": int(crop_top), "crop_left": int(crop_left),
        "target_h": int(th), "target_w": int(tw),
        "mode": mode, "value": float(value),
    }
    return x, pad_info
