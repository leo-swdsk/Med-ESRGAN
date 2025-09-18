import torch.nn.functional as F

def downsample_tensor(tensor, scale_factor=2, antialias=True):
    """
    Works for:
    - Single slices: [1, H, W]
    - Volume: [N, 1, H, W] (not used for now, but kept for future use (3D Superresolution))
    """
    if tensor.ndim == 3:
        # Single slice: [1, H, W] → [1, 1, H, W]
        tensor = tensor.unsqueeze(0)
        downsampled = F.interpolate(tensor, scale_factor=1/scale_factor, mode='bilinear', align_corners=False, antialias=antialias)
        return downsampled.squeeze(0)  # back to [1, H', W']

    elif tensor.ndim == 4: # Volume
        # Volume: [N, 1, H, W]
        downsampled = F.interpolate(tensor, scale_factor=1/scale_factor, mode='bilinear', align_corners=False, antialias=antialias)
        return downsampled  # [N, 1, H', W']

    else:
        raise ValueError("Tensor must be [1,H,W] or [N,1,H,W]")
