import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pydicom.pixel_data_handlers.util import apply_modality_lut

from window_presets import WINDOW_PRESETS



def _gaussian_kernel_2d(sigma: float, kernel_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    half = (k - 1) // 2
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    g1 = torch.exp(-(x ** 2) / (2 * sigma * sigma))
    g1 = g1 / g1.sum()
    kernel = torch.outer(g1, g1)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, k, k)


def _kernel_size_from_sigma(sigma: float) -> int:
    k = int(max(3, round(6.0 * float(sigma))))
    if k % 2 == 0:
        k += 1
    return k


def degrade_hr_to_lr(hr_volume: torch.Tensor, scale: int, *, degradation: str = 'blurnoise', blur_sigma_range=None,
                     blur_kernel: int = None, noise_sigma_range_norm=(0.001, 0.003), dose_factor_range=(0.25, 0.5),
                     antialias_clean: bool = True) -> torch.Tensor:
    # hr_volume: [D,1,H,W]
    device = hr_volume.device
    dtype = hr_volume.dtype
    if degradation in ('blur', 'blurnoise'):
        # sample sigma per-call (visualization: one sigma for the whole volume to keep coherent look)
        if blur_sigma_range is None:
            base_sigma = 0.8 if scale == 2 else (1.2 if scale == 4 else 0.8)
            jitter = 0.1 if scale == 2 else 0.15
            sig_lo, sig_hi = max(1e-6, base_sigma - jitter), base_sigma + jitter
        else:
            sig_lo, sig_hi = float(blur_sigma_range[0]), float(blur_sigma_range[1])
        sigma = float(np.random.default_rng().uniform(sig_lo, sig_hi))
        k = blur_kernel if blur_kernel is not None else _kernel_size_from_sigma(sigma)
        kernel = _gaussian_kernel_2d(max(1e-6, sigma), k, device, dtype)
        pad = (k // 2, k // 2, k // 2, k // 2)
        x = F.pad(hr_volume, pad, mode='reflect')
        hr_blur = F.conv2d(x, kernel)
    else:
        hr_blur = hr_volume

    # downsample
    if degradation == 'clean':
        lr = F.interpolate(hr_blur, scale_factor=(1.0/scale, 1.0/scale), mode='bilinear', align_corners=False, antialias=antialias_clean)
    else:
        lr = F.interpolate(hr_blur, scale_factor=(1.0/scale, 1.0/scale), mode='bilinear', align_corners=False, antialias=False)

    # noise
    if degradation == 'blurnoise':
        n_lo, n_hi = float(noise_sigma_range_norm[0]), float(noise_sigma_range_norm[1])
        d_lo, d_hi = float(dose_factor_range[0]), float(dose_factor_range[1])
        rng = np.random.default_rng()
        noise_sigma = float(rng.uniform(n_lo, n_hi))
        dose = float(rng.uniform(min(d_lo, d_hi), max(d_lo, d_hi)))
        noise_eff = noise_sigma / max(1e-6, dose) ** 0.5
        noise_np = np.random.default_rng().normal(loc=0.0, scale=noise_eff, size=tuple(lr.shape))
        noise_t = torch.as_tensor(noise_np, device=lr.device, dtype=lr.dtype)
        lr = torch.clamp(lr + noise_t, -1.0, 1.0)
    return lr



def load_ct_volume(folder_path, preset="soft_tissue"):
    # Delegate to centralized loader for consistency
    from ct_series_loader import load_series_windowed
    return load_series_windowed(folder_path, preset=preset)


def to_display(img_tensor):
    img = img_tensor.detach().cpu().numpy()
    img = ((img + 1.0) / 2.0).clip(0.0, 1.0)
    return img


def extract_slice(volume, index):
    D, _, _, _ = volume.shape
    index = int(np.clip(index, 0, D - 1))
    return volume[index, 0, :, :], D, index


def map_index_between_hr_lr(hr_index, hr_shape, lr_shape):
    D_hr, _, _, _ = hr_shape
    D_lr, _, _, _ = lr_shape
    return int(np.clip(hr_index, 0, min(D_hr, D_lr) - 1))


def build_lr_volume_from_hr(hr_volume, scale=2, *, degradation='blurnoise', blur_sigma_range=None, blur_kernel=None,
							 noise_sigma_range_norm=(0.001, 0.003), dose_factor_range=(1.0, 1.0), antialias_clean=True):
	return degrade_hr_to_lr(hr_volume, scale,
		degradation=degradation,
		blur_sigma_range=blur_sigma_range,
		blur_kernel=blur_kernel,
		noise_sigma_range_norm=noise_sigma_range_norm,
		dose_factor_range=dose_factor_range,
		antialias_clean=antialias_clean)


class ViewerLRHR:
    def __init__(self, lr_volume, hr_volume):
        self.lr = lr_volume   # [D,1,h,w]
        self.hr = hr_volume   # [D,1,H,W]
        D, _, _, _ = self.hr.shape
        self.index = D // 2

        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 6))
        self.ax_lr, self.ax_hr = self.axes
        self.ax_lr.set_title('LR')
        self.ax_hr.set_title('HR (original)')
        for ax in self.axes:
            ax.axis('off')

        self.im_lr = None
        self.im_hr = None
        self.text = self.fig.text(0.5, 0.02, '', ha='center', va='bottom')

        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.update()

    def update(self):
        D_hr, _, _, _ = self.hr.shape
        D_lr, _, _, _ = self.lr.shape
        clamped_idx = int(np.clip(self.index, 0, min(D_hr, D_lr) - 1))

        lr_plane, _, _ = extract_slice(self.lr, clamped_idx)
        hr_plane, axis_len, _ = extract_slice(self.hr, clamped_idx)

        img_lr = to_display(lr_plane)
        img_hr = to_display(hr_plane)

        if self.im_lr is None:
            self.im_lr = self.ax_lr.imshow(img_lr, cmap='gray', vmin=0, vmax=1)
        else:
            self.im_lr.set_data(img_lr)
        if self.im_hr is None:
            self.im_hr = self.ax_hr.imshow(img_hr, cmap='gray', vmin=0, vmax=1)
        else:
            self.im_hr.set_data(img_hr)

        self.text.set_text(f'Index: {clamped_idx+1}/{axis_len}')
        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        step = 1 if getattr(event, 'step', 0) >= 0 else -1
        if event.button == 'up':
            step = 1
        elif event.button == 'down':
            step = -1
        self.index += step
        self.update()


def main():
    parser = argparse.ArgumentParser(description='Visualize LR vs HR CT slices with mouse-wheel scrolling')
    parser.add_argument('--dicom_folder', type=str, required=True, help='Root folder containing DICOM series')
    parser.add_argument('--preset', type=str, default='soft_tissue', help='Window preset')
    parser.add_argument('--scale', type=int, default=2, help='Upsampling scale (must match model)')
    # Degradation flags (default blurnoise)
    parser.add_argument('--degradation', type=str, default='blurnoise', choices=['clean', 'blur', 'blurnoise'], help='Degradation pipeline for LR generation (default: blurnoise)')
    parser.add_argument('--blur_sigma_range', type=float, nargs=2, default=None, help='Range [lo hi] of Gaussian blur sigma; if None, defaults by scale')
    parser.add_argument('--blur_kernel', type=int, default=None, help='Explicit odd kernel size; if None, derived from sigma')
    parser.add_argument('--noise_sigma_range_norm', type=float, nargs=2, default=[0.001, 0.003], help='Gaussian noise sigma range on normalized [-1,1] image')
    parser.add_argument('--dose_factor_range', type=float, nargs=2, default=[0.25, 0.5], help='Dose factor range; noise scales ~ 1/sqrt(dose)')
    parser.add_argument('--antialias_clean', action='store_true', help='Use antialias in clean downsample')
    args = parser.parse_args()

    hr_vol = load_ct_volume(args.dicom_folder, preset=args.preset)          # [D,1,H,W]
    print(f"[Vis] Degradation='{args.degradation}' | blur_sigma_range={args.blur_sigma_range} | blur_kernel={args.blur_kernel} | noise_sigma_range_norm={args.noise_sigma_range_norm} | dose_factor_range={args.dose_factor_range}")
    lr_vol = build_lr_volume_from_hr(
        hr_vol, scale=args.scale,
        degradation=args.degradation,
        blur_sigma_range=args.blur_sigma_range,
        blur_kernel=args.blur_kernel,
        noise_sigma_range_norm=args.noise_sigma_range_norm,
        dose_factor_range=args.dose_factor_range,
        antialias_clean=args.antialias_clean
    )

    viewer = ViewerLRHR(lr_vol, hr_vol)
    print('Scroll with mouse wheel to navigate slices (LR left, HR right)')
    plt.show()


if __name__ == '__main__':
    main() 