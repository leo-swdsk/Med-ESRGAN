import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Set

try:
    import pyiqa as _pyiqa
except Exception:
    _pyiqa = None
try:
    import lpips as _lpips
except Exception:
    _lpips = None

from skimage.metrics import structural_similarity as sk_ssim


METRICS_CORE_VERSION = "0.1.0"

_lpips_models: Dict[Tuple[str, str], Tuple[torch.nn.Module, str]] = {}  # (device_str, backbone)->(model, backend)
_piqa_niqe_by_device = {}    # device->metric
_piqa_ma_by_device = {}      # device->metric
_lpips_input_warned = False
_lpips_logged: Set[Tuple[str, str]] = set()


def preprocess_for_metrics(img_hu: torch.Tensor, *, mode: str = 'global', hu_clip: Tuple[float, float] = (-1000.0, 2000.0), wl: Optional[float] = None, ww: Optional[float] = None, out_range: str = '[0,1]') -> torch.Tensor:
    x = img_hu.to(torch.float32)
    if mode == 'global':
        lo, hi = float(hu_clip[0]), float(hu_clip[1])
        x = torch.clamp(x, lo, hi)
        x01 = (x - lo) / (hi - lo)
    elif mode == 'window':
        assert wl is not None and ww is not None, "window mode requires wl and ww"
        min_val = float(wl) - float(ww) / 2.0
        max_val = float(wl) + float(ww) / 2.0
        x = torch.clamp(x, min_val, max_val)
        x01 = (x - min_val) / (max_val - min_val)
    else:
        raise ValueError("mode must be 'global' or 'window'")
    if out_range == '[0,1]':
        return x01.clamp(0.0, 1.0).to(torch.float32)
    raise ValueError("out_range must be '[0,1]'")


def align_and_crop(sr: torch.Tensor, hr: torch.Tensor, *, require_same_shape: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    Hs, Ws = sr.shape[-2:]
    Hh, Wh = hr.shape[-2:]
    if (Hs, Ws) == (Hh, Wh):
        return sr, hr
    Hc, Wc = min(Hs, Hh), min(Ws, Wh)
    def cc(x):
        Hx, Wx = x.shape[-2:]
        y0 = max(0, (Hx - Hc) // 2)
        x0 = max(0, (Wx - Wc) // 2)
        return x[..., y0:y0+Hc, x0:x0+Wc]
    sr_c, hr_c = cc(sr), cc(hr)
    if require_same_shape:
        assert sr_c.shape[-2:] == hr_c.shape[-2:], "align_and_crop failed to produce same shape"
    return sr_c, hr_c


def compute_psnr(sr01: torch.Tensor, hr01: torch.Tensor, *, max_val: float = 1.0) -> float:
    diff = (hr01 - sr01).to(torch.float32)
    mse = torch.mean(diff * diff).item()
    if mse <= 0.0:
        return float('inf')
    return float(10.0 * math.log10((max_val * max_val) / mse))


def compute_ssim(sr01: torch.Tensor, hr01: torch.Tensor, *, data_range: float = 1.0, win_size: int = 11, gaussian_weights: bool = True, sigma: float = 1.5, K1: float = 0.01, K2: float = 0.03) -> float:
    # Convert to numpy 2D arrays
    a = sr01.squeeze().detach().cpu().numpy()
    b = hr01.squeeze().detach().cpu().numpy()
    # Ensure odd win_size and >=3 and <= min(H,W)
    h, w = a.shape
    ws = min(win_size, h, w)
    if ws % 2 == 0:
        ws = max(3, ws - 1)
    ws = max(3, ws)
    try:
        val = sk_ssim(b, a, data_range=data_range, win_size=ws, gaussian_weights=gaussian_weights, sigma=sigma, use_sample_covariance=False, K1=K1, K2=K2)
        return float(val)
    except Exception:
        return 0.0


def _ensure_lpips(backbone: str, device: torch.device):
    key = (str(device), backbone)
    cached = _lpips_models.get(key)
    if cached is not None:
        return cached
    # Prefer pyiqa
    if _pyiqa is not None:
        try:
            m = _pyiqa.create_metric('lpips', device=str(device), net=backbone)
            m.eval()
            _lpips_models[key] = (m, 'pyiqa')
            return _lpips_models[key]
        except Exception:
            pass
    # Fallback lpips pkg
    if _lpips is not None:
        try:
            m = _lpips.LPIPS(net=backbone).to(device)
            m.eval()
            _lpips_models[key] = (m, 'lpips')
            return _lpips_models[key]
        except Exception:
            pass
    return None, 'none'


def compute_lpips(sr01: torch.Tensor, hr01: torch.Tensor, *, backbone: str = 'alex', device: str = 'cuda') -> float:
    dev_req = torch.device(device)
    model, backend = _ensure_lpips(backbone, dev_req)
    if model is None:
        return float('nan')

    # Guard: clamp inputs to [0,1] if slightly off, warn only once
    global _lpips_input_warned
    def _maybe_warn(name: str, t: torch.Tensor):
        global _lpips_input_warned
        tmin = float(torch.min(t).item()) if t.numel() > 0 else 0.0
        tmax = float(torch.max(t).item()) if t.numel() > 0 else 0.0
        if (tmin < -1e-6 or tmax > 1.0 + 1e-6) and (not _lpips_input_warned):
            print(f"[LPIPS] input clamped for {name}: min={tmin:.4f} max={tmax:.4f}")
            _lpips_input_warned = True

    _maybe_warn('sr01', sr01)
    _maybe_warn('hr01', hr01)
    sr01 = sr01.to(torch.float32).clamp(0.0, 1.0)
    hr01 = hr01.to(torch.float32).clamp(0.0, 1.0)

    # Helpers per spec
    def to01_3(x: torch.Tensor, dev: torch.device) -> torch.Tensor:
        x = x.to(torch.float32).clamp(0.0, 1.0)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        return x.repeat(1, 3, 1, 1).to(dev, non_blocking=True)

    def to_m11_3(x: torch.Tensor, dev: torch.device) -> torch.Tensor:
        x = x.to(torch.float32).clamp(0.0, 1.0)
        x = x * 2.0 - 1.0
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        return x.repeat(1, 3, 1, 1).to(dev, non_blocking=True)

    with torch.no_grad():
        if backend == 'pyiqa':
            try:
                dev = next(model.parameters()).device
            except Exception:
                dev = dev_req
            xs = to01_3(sr01, dev)
            xh = to01_3(hr01, dev)
            # one-time debug log
            key = (str(dev), backbone)
            if key not in _lpips_logged:
                print(f"[LPIPS] backend={backend} backbone={backbone} device={dev}")
                _lpips_logged.add(key)
            # debug self-test
            xmin = float(xs.min().item()); xmax = float(xs.max().item())
            assert xmin >= -1e-6 and xmax <= 1.0 + 1e-6, f"pyiqa xs out of [0,1]: {xmin},{xmax}"
            d = model(xs, xh)
            return float(d.item())
        else:
            # lpips package
            try:
                dev = next(model.parameters()).device
            except Exception:
                dev = torch.device('cpu')
            xs = to_m11_3(sr01, dev)
            xh = to_m11_3(hr01, dev)
            # one-time debug log
            key = (str(dev), backbone)
            if key not in _lpips_logged:
                print(f"[LPIPS] backend={backend} backbone={backbone} device={dev}")
                _lpips_logged.add(key)
            # debug self-test
            xmin = float(xs.min().item()); xmax = float(xs.max().item())
            assert xmin >= -1.0 - 1e-6 and xmax <= 1.0 + 1e-6, f"lpips xs out of [-1,1]: {xmin},{xmax}"
            d = model(xs, xh)
            return float(d.item())


def _ensure_pi_metrics(device: torch.device):
    dev_key = str(device)
    niqe = _piqa_niqe_by_device.get(dev_key)
    ma = _piqa_ma_by_device.get(dev_key)
    if (niqe is None or ma is None) and _pyiqa is not None:
        try:
            if niqe is None:
                niqe = _pyiqa.create_metric('niqe', device=dev_key)
                niqe.eval()
                _piqa_niqe_by_device[dev_key] = niqe
            if ma is None:
                ma = _pyiqa.create_metric('nrqm', device=dev_key)
                ma.eval()
                _piqa_ma_by_device[dev_key] = ma
        except Exception:
            pass
    return _piqa_niqe_by_device.get(dev_key), _piqa_ma_by_device.get(dev_key)


def compute_pi(sr01_gray: torch.Tensor, *, device: str = 'cpu') -> Tuple[float, float, float]:
    dev = torch.device(device)
    niqe_m, ma_m = _ensure_pi_metrics(dev)

    # Ensure [B,1,H,W] in [0,1]
    x = sr01_gray.to(torch.float32)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x = x.clamp(0.0, 1.0)

    niqe_val = float('nan')
    ma_val = float('nan')

    # NIQE on grayscale at original size; upscale to min edge 96 if smaller
    try:
        if niqe_m is not None:
            H, W = x.shape[-2:]
            x_niqe = x
            if min(H, W) < 96:
                s = 96.0 / float(min(H, W))
                newH, newW = int(round(H * s)), int(round(W * s))
                x_niqe = F.interpolate(x, size=(newH, newW), mode='bilinear', align_corners=False)
            with torch.no_grad():
                niqe_val = float(niqe_m(x_niqe.to(dev)).item())
    except Exception:
        try:
            from skimage.metrics import niqe as _sk_niqe
            niqe_val = float(_sk_niqe(x.squeeze(0).squeeze(0).detach().cpu().numpy().astype('float64')))
        except Exception:
            pass

    # Ma/NRQM on 3ch resized 224x224
    try:
        if ma_m is not None:
            x3 = x.repeat(1, 3, 1, 1)
            x3 = F.interpolate(x3, size=(224, 224), mode='bilinear', align_corners=False)
            with torch.no_grad():
                ma_val = float(ma_m(x3.to(dev)).item())
    except Exception:
        pass

    pi_val = float('nan')
    if math.isfinite(niqe_val) and math.isfinite(ma_val):
        pi_val = float(0.5 * ((10.0 - ma_val) + niqe_val))
    return pi_val, ma_val, niqe_val


def compute_all_metrics(sr_hu: torch.Tensor, hr_hu: torch.Tensor, *, mode: str = 'global', hu_clip: Tuple[float, float] = (-1000.0, 2000.0), wl: Optional[float] = None, ww: Optional[float] = None, lpips_backbone: str = 'alex', device: str = 'cuda', return_components: bool = True) -> Dict[str, float]:
    # Inputs ideally HU. Guard against normalized inputs and handle gracefully.
    sr_hu = sr_hu.to(torch.float32)
    hr_hu = hr_hu.to(torch.float32)
    smin = float(sr_hu.min().item()) if sr_hu.numel() > 0 else 0.0
    smax = float(sr_hu.max().item()) if sr_hu.numel() > 0 else 0.0
    hmin = float(hr_hu.min().item()) if hr_hu.numel() > 0 else 0.0
    hmax = float(hr_hu.max().item()) if hr_hu.numel() > 0 else 0.0

    def _warn_once(msg: str):
        # simple print; user can search logs
        print(msg)

    # Decide domain handling
    if (smax <= 1.2 and smin >= -0.2) and (hmax <= 1.2 and hmin >= -0.2):
        # Likely [0,1] already (viewer/evaluator preprocessed). Use as-is after clamp and warn.
        _warn_once("[Metrics] WARNING: expected HU but received [0,1]; skipping HU preprocessing. Check call site.")
        sr01 = sr_hu.clamp(0.0, 1.0)
        hr01 = hr_hu.clamp(0.0, 1.0)
    elif (smax <= 1.05 and smin >= -1.05) and (hmax <= 1.05 and hmin >= -1.05):
        # Likely [-1,1] tensors → map to [0,1]
        _warn_once("[Metrics] WARNING: expected HU but received [-1,1]; mapping to [0,1] and skipping HU preprocessing. Check call site.")
        sr01 = ((sr_hu + 1.0) / 2.0).clamp(0.0, 1.0)
        hr01 = ((hr_hu + 1.0) / 2.0).clamp(0.0, 1.0)
    else:
        # Treat as HU → preprocess to [0,1]
        sr01 = preprocess_for_metrics(sr_hu, mode=mode, hu_clip=hu_clip, wl=wl, ww=ww)
        hr01 = preprocess_for_metrics(hr_hu, mode=mode, hu_clip=hu_clip, wl=wl, ww=ww)
    # Align shapes (center crop)
    sr01, hr01 = align_and_crop(sr01, hr01, require_same_shape=True)
    # Add channel dims [1,H,W]
    if sr01.ndim == 2:
        sr01 = sr01.unsqueeze(0)
    if hr01.ndim == 2:
        hr01 = hr01.unsqueeze(0)

    # Metrics
    mae = torch.mean(torch.abs(hr01 - sr01)).item()
    rmse = math.sqrt(torch.mean((hr01 - sr01) ** 2).item())
    psnr = compute_psnr(sr01, hr01, max_val=1.0)
    ssim = compute_ssim(sr01, hr01, data_range=1.0, win_size=11, gaussian_weights=True, sigma=1.5, K1=0.01, K2=0.03)
    lpips = compute_lpips(sr01, hr01, backbone=lpips_backbone, device=device)
    pi, ma, niqe = compute_pi(sr01, device=device)

    out = {
        'MSE': float(rmse * rmse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'PSNR': float(psnr),
        'SSIM': float(ssim),
        'LPIPS': float(lpips),
        'MA': float(ma),
        'NIQE': float(niqe),
        'PI': float(pi),
    }
    # Attach version/config meta keys convention
    out['_meta'] = {
        'metrics_core_version': METRICS_CORE_VERSION,
        'lpips_backbone': lpips_backbone,
        'scale_mode': mode,
        'ssim_params': {'win_size': 11, 'gaussian_weights': True, 'sigma': 1.5, 'K1': 0.01, 'K2': 0.03},
    }
    return out


def debug_compare(slice_id: int, sr_hu: torch.Tensor, hr_hu: torch.Tensor, **kwargs) -> None:
    # Prepare preprocessed [0,1]
    mode = kwargs.get('mode', 'global')
    hu_clip = kwargs.get('hu_clip', (-1000.0, 2000.0))
    wl = kwargs.get('wl', None)
    ww = kwargs.get('ww', None)
    device = kwargs.get('device', 'cpu')
    backbone = kwargs.get('lpips_backbone', 'alex')
    sr01 = preprocess_for_metrics(sr_hu, mode=mode, hu_clip=hu_clip, wl=wl, ww=ww)
    hr01 = preprocess_for_metrics(hr_hu, mode=mode, hu_clip=hu_clip, wl=wl, ww=ww)
    sr01, hr01 = align_and_crop(sr01, hr01, require_same_shape=True)

    # Backend-specific transform sanity checks
    model, backend = _ensure_lpips(backbone, torch.device(device))
    if model is not None:
        # same helpers as compute_lpips
        def to01_3(x: torch.Tensor, dev: torch.device) -> torch.Tensor:
            x = x.to(torch.float32).clamp(0.0, 1.0)
            if x.ndim == 2:
                x = x.unsqueeze(0)
            if x.ndim == 3:
                x = x.unsqueeze(0)
            return x.repeat(1, 3, 1, 1).to(dev, non_blocking=True)
        def to_m11_3(x: torch.Tensor, dev: torch.device) -> torch.Tensor:
            x = x.to(torch.float32).clamp(0.0, 1.0)
            x = x * 2.0 - 1.0
            if x.ndim == 2:
                x = x.unsqueeze(0)
            if x.ndim == 3:
                x = x.unsqueeze(0)
            return x.repeat(1, 3, 1, 1).to(dev, non_blocking=True)
        try:
            devm = next(model.parameters()).device
        except Exception:
            devm = torch.device('cpu')
        if backend == 'pyiqa':
            xs = to01_3(sr01, devm)
            xmin = float(xs.min().item()); xmax = float(xs.max().item())
            assert xmin >= -1e-6 and xmax <= 1.0 + 1e-6, f"pyiqa xs out of [0,1]: {xmin},{xmax}"
        else:
            xs = to_m11_3(sr01, devm)
            xmin = float(xs.min().item()); xmax = float(xs.max().item())
            assert xmin >= -1.0 - 1e-6 and xmax <= 1.0 + 1e-6, f"lpips xs out of [-1,1]: {xmin},{xmax}"

    m = compute_all_metrics(sr_hu, hr_hu, mode=mode, hu_clip=hu_clip, wl=wl, ww=ww, lpips_backbone=backbone, device=device, return_components=True)
    print(f"[debug_compare] slice={slice_id} | PSNR={m['PSNR']:.4f} SSIM={m['SSIM']:.4f} LPIPS={m['LPIPS']:.4f} PI={m['PI']:.4f} | MA={m['MA']:.4f} NIQE={m['NIQE']:.4f}")


