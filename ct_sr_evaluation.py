import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import math
from metrics_core import compute_all_metrics

# Optional metrics: LPIPS (full-reference) and Perceptual Index (PI = 0.5*((10-Ma)+NIQE))
_lpips_model = None  # fallback lpips package model (moved to current device lazily)
_pyiqa_lpips_by_device = {}  # device(str)->metric
_piqa_niqe_by_device = {}
_piqa_ma_by_device = {}
_warned_lpips = False
_warned_pi = False
try:
    import lpips as _lpips
except Exception:
    _lpips = None
try:
    import pyiqa as _pyiqa
except Exception:
    _pyiqa = None
try:
    from skimage.metrics import niqe as _skimage_niqe
except Exception:
    _skimage_niqe = None

def upsample_interpolation(lr_tensor, target_size, method="bilinear"):
    # lr_tensor: [1,h,w] in [-1,1]; target_size: (H, W)
    mode = "bilinear" if method == "bilinear" else "bicubic"
    x = lr_tensor.unsqueeze(0)  # [1,1,h,w]
    y = F.interpolate(x, size=target_size, mode=mode, align_corners=False)
    return y.squeeze(0)

def _ensure_lpips_model(target_device: torch.device):
    """Prefer pyiqa's LPIPS (newer API), fallback to lpips package. Creates per-device instance."""
    global _pyiqa_lpips_by_device, _lpips_model, _warned_lpips
    dev_key = str(target_device)
    if _pyiqa is not None:
        metric = _pyiqa_lpips_by_device.get(dev_key)
        if metric is None:
            try:
                # Explicit net and device to avoid implicit downloads/device mismatch
                metric = _pyiqa.create_metric('lpips', device=dev_key, net='alex')
                metric.eval()
                _pyiqa_lpips_by_device[dev_key] = metric
            except Exception:
                metric = None
        if metric is not None:
            return metric
    # fallback to original lpips
    if _lpips_model is not None:
        # move to target device if necessary
        try:
            _lpips_model = _lpips_model.to(target_device)
        except Exception:
            pass
        return _lpips_model
    if _lpips is None:
        if not _warned_lpips:
            print("[Metrics] LPIPS not available (pip install pyiqa or lpips). LPIPS will be NaN.")
            _warned_lpips = True
        return None
    try:
        _lpips_model = _lpips.LPIPS(net='alex')
        try:
            _lpips_model = _lpips_model.to(target_device)
        except Exception:
            pass
        _lpips_model.eval()
    except Exception:
        _lpips_model = None
        if not _warned_lpips:
            print("[Metrics] Failed to initialize LPIPS(alex). LPIPS will be NaN.")
            _warned_lpips = True
    return _lpips_model

def _compute_lpips(sr_tensor, hr_tensor):
    # Inputs: [1,H,W] in [-1,1]
    # Select target device based on input tensors (prefer CUDA if any input is on CUDA)
    target_device = sr_tensor.device if sr_tensor.is_cuda else (hr_tensor.device if hr_tensor.is_cuda else torch.device('cpu'))
    model = _ensure_lpips_model(target_device)
    if model is None:
        print("[LPIPS] model not available; returning NaN")
        return float('nan')
    try:
        with torch.no_grad():
            if _pyiqa is not None and str(type(model)).find('pyiqa') != -1:
                # pyiqa expects [B,C,H,W] in [0,1] (it will internally normalize)
                def to01_3(x):
                    x01 = torch.clamp(x.float(), -1.0, 1.0)
                    x01 = (x01 + 1.0) * 0.5
                    x3 = x01.unsqueeze(0).repeat(1,3,1,1).contiguous()  # [1,3,H,W]
                    return x3
                x_s = to01_3(sr_tensor).to(target_device)
                x_h = to01_3(hr_tensor).to(target_device)
                d = model(x_s, x_h)
                val = float(d.item())
                print(f"[LPIPS] backend=pyiqa value={val:.4f}")
            else:
                # lpips package expects [-1,1] 3ch
                def to_m11_3(x):
                    x_m11 = torch.clamp(x.float(), -1.0, 1.0)
                    x3 = x_m11.repeat(3, 1, 1)
                    return x3.unsqueeze(0).contiguous()
                xs = to_m11_3(sr_tensor).to(target_device)
                xh = to_m11_3(hr_tensor).to(target_device)
                d = model(xs, xh)
                val = float(d.item())
                backend = 'lpips-pkg' if _lpips is not None else 'unknown'
                print(f"[LPIPS] backend={backend} value={val:.4f}")
        return val
    except Exception:
        print("[LPIPS] computation failed; returning NaN")
        return float('nan')

def _ensure_pi_metrics(target_device: torch.device):
    global _piqa_niqe_by_device, _piqa_ma_by_device, _warned_pi
    dev_key = str(target_device)
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
            # leave niqe/ma as is; may be None
            pass
    if (_piqa_niqe_by_device.get(dev_key) is None or _piqa_ma_by_device.get(dev_key) is None) and not _warned_pi:
        msg = "[Metrics] PI requires NIQE and Ma."
        if _pyiqa is None:
            msg += " Install pyiqa for NIQE/Ma (pip install pyiqa)."
        else:
            msg += " pyiqa not fully available; PI will be NaN."
        if _skimage_niqe is None:
            msg += " skimage.niqe also not available."
        print(msg)
        _warned_pi = True
    return _piqa_niqe_by_device.get(dev_key), _piqa_ma_by_device.get(dev_key)

def _compute_pi(grayscale_tensor_m1_1):
    pi_val, _, _ = _compute_pi_with_components(grayscale_tensor_m1_1)
    return pi_val

def _compute_pi_with_components(grayscale_tensor_m1_1):
    """
    Input: [1,H,W] in [-1,1]  (ein Slice, 1 Kanal)
    Return: (PI, MA, NIQE)
    """
    # target device after availability
    target_device = grayscale_tensor_m1_1.device if grayscale_tensor_m1_1.is_cuda else torch.device('cpu')
    piqa_niqe, piqa_ma = _ensure_pi_metrics(target_device)

    # [-1,1] -> [0,1], robust against dimensions
    x = grayscale_tensor_m1_1.detach().to(torch.float32)
    if x.ndim == 2:           # [H,W] -> [1,H,W]
        x = x.unsqueeze(0)
    assert x.ndim == 3, f"expected [1,H,W], got {tuple(x.shape)}"
    x01 = ((x + 1.0) / 2.0).clamp(0.0, 1.0)  # [1,H,W]
    x01 = x01.unsqueeze(0)                   # -> [1,1,H,W]  (B=1,C=1,H,W)

    niqe_val = float('nan')
    ma_val   = float('nan')

    try:
        # -------- NIQE (1ch, Original size; <96 px softly upscale) --------
        if piqa_niqe is not None:
            H, W = x01.shape[-2:]
            if min(H, W) < 96:
                s = 96.0 / float(min(H, W))
                newH, newW = int(round(H*s)), int(round(W*s))
                x_niqe = torch.nn.functional.interpolate(
                    x01, size=(newH,newW), mode='bilinear', align_corners=False
                )
            else:
                x_niqe = x01
            dev_niqe = next(piqa_niqe.parameters()).device if hasattr(piqa_niqe, 'parameters') else torch.device('cpu')
            with torch.no_grad():
                niqe_val = float(piqa_niqe(x_niqe.to(dev_niqe)).item())

        elif _skimage_niqe is not None:
            # skimage expects 2D [H,W] float in [0,1]
            niqe_val = float(_skimage_niqe(x01.squeeze(0).squeeze(0).cpu().numpy().astype('float64')))

    except Exception as e:
        # optional: print(f"[NIQE] error: {type(e).__name__}: {e}")
        pass

    try:
        # -------- MA/NRQM (3ch, 224x224) --------
        if piqa_ma is not None:
            x_ma = x01.repeat(1, 3, 1, 1)  # [1,3,H,W]
            x_ma = torch.nn.functional.interpolate(x_ma, size=(224,224), mode='bilinear', align_corners=False)
            dev_ma = next(piqa_ma.parameters()).device if hasattr(piqa_ma, 'parameters') else torch.device('cpu')
            with torch.no_grad():
                ma_val = float(piqa_ma(x_ma.to(dev_ma)).item())

    except Exception as e:
        # optional: print(f"[MA/NRQM] error: {type(e).__name__}: {e}")
        pass

    pi_val = float('nan')
    if math.isfinite(niqe_val) and math.isfinite(ma_val):
        pi_val = float(0.5 * ((10.0 - ma_val) + niqe_val))

    print(f"[PI-Debug] NIQE={niqe_val if math.isfinite(niqe_val) else float('nan'):.4f} "
          f"MA={ma_val if math.isfinite(ma_val) else float('nan'):.4f} -> PI={pi_val if math.isfinite(pi_val) else float('nan'):.4f}")
    return pi_val, ma_val, niqe_val


def _denorm_global_to_hu(x_m11, hu_clip):
    # x_m11: [1,H,W] in [-1,1] normalized with global HU clip
    x01 = (x_m11.clamp(-1.0, 1.0) + 1.0) * 0.5
    lo, hi = float(hu_clip[0]), float(hu_clip[1])
    return x01 * (hi - lo) + lo

def evaluate_metrics(sr_tensor, hr_tensor, *, hu_clip=(-1000.0, 2000.0), metrics_mode='global', window_center=None, window_width=None, metrics_device='cpu'):
    # -> HU via global denormalization
    sr_hu = _denorm_global_to_hu(sr_tensor, hu_clip).squeeze(0)
    hr_hu = _denorm_global_to_hu(hr_tensor, hu_clip).squeeze(0)
    # metrics preprocessing mode
    if metrics_mode == 'window':
        mode = 'window'
        wl = float(window_center)
        ww = float(window_width)
    else:
        mode = 'global'
        wl = None
        ww = None
    m = compute_all_metrics(sr_hu, hr_hu,
                            mode=mode, wl=wl, ww=ww,
                            lpips_backbone='alex', device=metrics_device,
                            return_components=True)
    return {k: float(m[k]) for k in ['MSE','RMSE','MAE','PSNR','SSIM','LPIPS','MA','NIQE','PI']}

def compare_methods(lr_tensor, hr_tensor, model,
                    *, hu_clip=(-1000.0, 2000.0),
                    metrics_mode='global', window_center=None, window_width=None,
                    metrics_device='cpu'):
    model.eval()
    with torch.no_grad():
        # Ensure model input is on the same device as model
        device = next(model.parameters()).device
        lr_batched = lr_tensor.unsqueeze(0).to(device)
        sr_model = model(lr_batched).squeeze(0).cpu()

        # Interpolation Upscaling on CPU tensors
        target_size = hr_tensor.shape[1:]  # [H, W]
        sr_linear = upsample_interpolation(lr_tensor, target_size, method="bilinear")
        sr_cubic = upsample_interpolation(lr_tensor, target_size, method="bicubic")

        # Safety: if model SR size differs from HR due to odd dimensions, center-crop all to common size
        def cc_common(x, ref):
            Hx, Wx = x.shape[-2:]
            Hr, Wr = ref.shape[-2:]
            Hc, Wc = min(Hx, Hr), min(Wx, Wr)
            def cc(t):
                Ht, Wt = t.shape[-2:]
                y0 = max(0, (Ht - Hc) // 2)
                x0 = max(0, (Wt - Wc) // 2)
                return t[..., y0:y0+Hc, x0:x0+Wc]
            return cc(x), cc(ref)

        sr_m, hr_m = (sr_model, hr_tensor)
        if sr_model.shape[-2:] != hr_tensor.shape[-2:]:
            sr_m, hr_m = cc_common(sr_model, hr_tensor)
        # Align interpolations to hr_m as well
        if sr_linear.shape[-2:] != hr_m.shape[-2:]:
            sr_linear, _ = cc_common(sr_linear, hr_m)
        if sr_cubic.shape[-2:] != hr_m.shape[-2:]:
            sr_cubic, _ = cc_common(sr_cubic, hr_m)

        # Compute metrics
        results = {
        "Model (RRDB)": evaluate_metrics(sr_m, hr_m, hu_clip=hu_clip, metrics_mode=metrics_mode, window_center=window_center, window_width=window_width, metrics_device=metrics_device),
        "Interpolation (Bilinear)": evaluate_metrics(sr_linear, hr_m, hu_clip=hu_clip, metrics_mode=metrics_mode, window_center=window_center, window_width=window_width, metrics_device=metrics_device),
        "Interpolation (Bicubic)": evaluate_metrics(sr_cubic, hr_m, hu_clip=hu_clip, metrics_mode=metrics_mode, window_center=window_center, window_width=window_width, metrics_device=metrics_device),
    }
    return results

