import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pydicom
import math
from pydicom.pixel_data_handlers.util import apply_modality_lut

from window_presets import WINDOW_PRESETS
from seed_utils import fixed_seed_for_path
from ct_dataset_loader import is_ct_image_dicom
from rrdb_ct_model import RRDBNet_CT
from skimage.metrics import structural_similarity as ssim
import io, contextlib
from metrics_core import compute_all_metrics
from ct_series_loader import load_series_hu, load_series_windowed
from windowing import apply_window



def read_series_metadata(folder_path):
    row_mm = None; col_mm = None; slice_thickness = None; patient_id = ''
    for root, _, files in os.walk(folder_path):
        for f in sorted(files):
            if not f.lower().endswith('.dcm'):
                continue
            path = os.path.join(root, f)
            if not is_ct_image_dicom(path):
                continue
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
                ps = getattr(ds, 'PixelSpacing', None)
                if ps is not None and len(ps) >= 2:
                    row_mm = float(ps[0]); col_mm = float(ps[1])
                st = getattr(ds, 'SliceThickness', None)
                if st is not None:
                    slice_thickness = float(st)
                patient_id = str(getattr(ds, 'PatientID', ''))
                return (row_mm, col_mm), slice_thickness, patient_id
            except Exception:
                continue
    return (row_mm, col_mm), slice_thickness, patient_id

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
    # Beide Volumen haben die gleiche Anzahl Schichten, Index 0-basiert
    return int(np.clip(hr_index, 0, min(D_hr, D_lr) - 1))


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
                     antialias_clean: bool = True, rng=None) -> tuple:
    device = hr_volume.device
    dtype = hr_volume.dtype
    used = {'blur_sigma': None, 'blur_kernel_k': None, 'noise_sigma': None, 'dose': None}
    if degradation in ('blur', 'blurnoise'):
        if blur_sigma_range is None:
            base_sigma = 0.8 if scale == 2 else (1.2 if scale == 4 else 0.8)
            jitter = 0.1 if scale == 2 else 0.15
            sig_lo, sig_hi = max(1e-6, base_sigma - jitter), base_sigma + jitter
        else:
            sig_lo, sig_hi = float(blur_sigma_range[0]), float(blur_sigma_range[1])
        rng = np.random.default_rng() if rng is None else rng
        sigma = float(rng.uniform(sig_lo, sig_hi))
        k = blur_kernel if blur_kernel is not None else _kernel_size_from_sigma(sigma)
        kernel = _gaussian_kernel_2d(max(1e-6, sigma), k, device, dtype)
        used['blur_sigma'] = float(sigma)
        used['blur_kernel_k'] = int(k)
        pad = (k // 2, k // 2, k // 2, k // 2)
        x = F.pad(hr_volume, pad, mode='reflect')
        hr_blur = F.conv2d(x, kernel)
    else:
        hr_blur = hr_volume

    if degradation == 'clean':
        lr = F.interpolate(hr_blur, scale_factor=(1.0/scale, 1.0/scale), mode='bilinear', align_corners=False, antialias=antialias_clean)
    else:
        lr = F.interpolate(hr_blur, scale_factor=(1.0/scale, 1.0/scale), mode='bilinear', align_corners=False, antialias=False)

    if degradation == 'blurnoise':
        n_lo, n_hi = float(noise_sigma_range_norm[0]), float(noise_sigma_range_norm[1])
        d_lo, d_hi = float(dose_factor_range[0]), float(dose_factor_range[1])
        rng = np.random.default_rng() if rng is None else rng
        noise_sigma = float(rng.uniform(n_lo, n_hi))
        dose = float(rng.uniform(min(d_lo, d_hi), max(d_lo, d_hi)))
        noise_eff = noise_sigma / max(1e-6, dose) ** 0.5
        # GPU-friendly Gaussian noise
        noise_t = torch.randn_like(lr, device=lr.device, dtype=lr.dtype) * noise_eff
        lr = torch.clamp(lr + noise_t, -1.0, 1.0)
        used['noise_sigma'] = float(noise_sigma)
        used['dose'] = float(dose)
    return lr, used


def build_lr_volume_from_hr(hr_volume, scale=2, *, degradation='blurnoise', blur_sigma_range=None, blur_kernel=None,
							 noise_sigma_range_norm=(0.001, 0.003), dose_factor_range=(0.25, 0.5), antialias_clean=True, rng=None):
	return degrade_hr_to_lr(hr_volume, scale,
		degradation=degradation,
		blur_sigma_range=blur_sigma_range,
		blur_kernel=blur_kernel,
		noise_sigma_range_norm=noise_sigma_range_norm,
		dose_factor_range=dose_factor_range,
		antialias_clean=antialias_clean,
		rng=rng)


def center_crop_to_multiple(volume: torch.Tensor, scale: int) -> torch.Tensor:
    # volume: [D,1,H,W]; crop H,W to nearest lower multiple of scale (centered)
    D, C, H, W = volume.shape
    target_H = (H // scale) * scale
    target_W = (W // scale) * scale
    if target_H <= 0 or target_W <= 0:
        return volume
    if target_H == H and target_W == W:
        return volume
    y0 = (H - target_H) // 2
    x0 = (W - target_W) // 2
    cropped = volume[:, :, y0:y0+target_H, x0:x0+target_W]
    print(f"[Vis] Center-cropped HR from ({H},{W}) -> ({target_H},{target_W}) to match scale={scale}")
    return cropped


def build_sr_volume_from_lr(lr_volume, model, batch_size: int = 8):
	device = next(model.parameters()).device
	model.eval()
	slices = lr_volume.shape[0]
	outs = []
	with torch.no_grad():
		use_cuda = (device.type == 'cuda')
		for i in range(0, slices, max(1, batch_size)):
			batch = lr_volume[i:i+batch_size]  # [B,1,h,w]
			with torch.amp.autocast('cuda', enabled=use_cuda):
				y = model(batch.to(device))  # [B,1,H,W]
			outs.append(y.cpu())
	return torch.cat(outs, dim=0)  # [D,1,H,W]

# ---------- Helper: HU <-> [-1,1] (global) & Anzeige-Fensterung ----------
def hu_to_m11_global(x, lo=-1000.0, hi=2000.0):
    x = x.to(torch.float32).clamp(lo, hi)
    x01 = (x - lo) / (hi - lo)
    return x01 * 2.0 - 1.0  # [-1,1]

def m11_to_hu_global(x, lo=-1000.0, hi=2000.0):
    x = x.to(torch.float32)
    x01 = (x.clamp(-1.0, 1.0) + 1.0) * 0.5
    return x01 * (hi - lo) + lo  # HU

def hu_to_m11_window(x: torch.Tensor, wl: float, ww: float) -> torch.Tensor:
    """Fenstert HU -> [-1,1] für die Anzeige (ohne Neuberechnung der SR)."""
    x = x.to(torch.float32)
    ww = float(ww)
    wl = float(wl)
    if ww <= 0:
        raise ValueError(f"Window Width must be > 0, got {ww}")
    min_val = wl - ww / 2.0
    max_val = wl + ww / 2.0
    x = x.clamp(min_val, max_val)
    x01 = (x - min_val) / (max_val - min_val)
    return x01 * 2.0 - 1.0  # [-1,1]

class ViewerLRSRHR:
    def __init__(self, lr_volume, sr_volume, hr_volume, scale=2, lin_volume=None, bic_volume=None, *,
                 dicom_folder=None, preset_name="soft_tissue", model=None, device=None,
                 pixel_spacing_mm=None, slice_thickness_mm=None, patient_id='',
                 hr_hu_volume=None, sr_hu_volume=None, lin_hu_volume=None, bic_hu_volume=None,
                 lr_hu_volume=None, slice_meta=None,
                 degradation=None, blur_sigma_range=None, blur_kernel=None, noise_sigma_range_norm=None,
                 dose_factor_range=None, antialias_clean=False, degradation_sampling=None, deg_seed=None): 
        self.lr = lr_volume
        self.sr = sr_volume
        self.hr = hr_volume
        self.lin = lin_volume
        self.bic = bic_volume
        # HU volumes for metrics
        self.lr_hu = lr_hu_volume
        self.hr_hu = hr_hu_volume
        self.sr_hu = sr_hu_volume
        self.lin_hu = lin_hu_volume
        self.bic_hu = bic_hu_volume
        self.scale = scale
        self.dicom_folder = dicom_folder
        self.preset_name = preset_name
        self.model = model
        self.device = device
        # metadata
        self.ps_row_mm = None if pixel_spacing_mm is None else pixel_spacing_mm[0]
        self.ps_col_mm = None if pixel_spacing_mm is None else pixel_spacing_mm[1]
        self.slice_thickness_mm = slice_thickness_mm
        self.patient_id = patient_id
        # optional slice meta for identity logging
        self.slice_meta = slice_meta if isinstance(slice_meta, list) else None
        # degradation config (for info panel)
        self.degradation = degradation
        self.blur_sigma_range = blur_sigma_range
        self.blur_kernel = blur_kernel
        self.noise_sigma_range_norm = noise_sigma_range_norm
        self.dose_factor_range = dose_factor_range
        self.antialias_clean = bool(antialias_clean)
        self.degradation_sampling = degradation_sampling
        self.deg_seed = deg_seed
        # effective degradation parameters (filled by caller; used for display only)
        self.used_sigma = None
        self.used_noise_sigma = None
        self.used_dose = None
        # track current windowing
        cfg = WINDOW_PRESETS.get(self.preset_name, WINDOW_PRESETS['default'])
        self.curr_wl = float(cfg['center'])
        self.curr_ww = float(cfg['width'])
        D_hr, _, _, _ = (self.hr_hu.shape if self.hr_hu is not None else self.hr.shape)
        self.index = 0  # Start bei Index 0 (inferior/lowest z)
        sr_info = ('lazy' if self.sr is None else str(tuple(self.sr.shape)))
        hr_info = (str(tuple(self.hr_hu.shape)) if self.hr_hu is not None else str(tuple(self.hr.shape)))
        print(f"[Viewer] init: D={D_hr} | LR={tuple(self.lr.shape)} SR={sr_info} HR={hr_info}")
        # simple lazy caches for per-slice SR/Interpolations
        self._sr_cache = {}
        self._lin_cache = {}
        self._bic_cache = {}

        # Axes: LR | Linear | Bicubic | SR | HR
        self.fig, self.axes = plt.subplots(1, 5, figsize=(22, 6))
        self.ax_lr, self.ax_lin, self.ax_bic, self.ax_sr, self.ax_hr = self.axes
        self.ax_lr.set_title('LR')
        self.ax_lin.set_title('Bilinear x{}'.format(scale))
        self.ax_bic.set_title('Bicubic x{}'.format(scale))
        self.ax_sr.set_title('SR (model)')
        self.ax_hr.set_title('HR')
        for ax in self.axes:
            ax.axis('off')
            ax.set_aspect('equal')

        self.im_lr = None
        self.im_lin = None
        self.im_bic = None
        self.im_sr = None
        self.im_hr = None
        # move index counter further right to avoid overlap with Index/Go controls
        self.text = self.fig.text(0.70, 0.02, '', ha='center', va='bottom')
        self.text_stats = self.fig.text(0.5, 0.97, '', ha='center', va='top')
        # info top, metrics will be placed below with spacing
        self.text_info = self.fig.text(0.02, 0.995, '', ha='left', va='top', family='monospace')
        self.text_roi = self.fig.text(0.02, 0.10, '', ha='left', va='top', family='monospace')
        self.metric_texts = []

        self.roi = None  # (x0,y0,x1,y1) in HR coordinate space
        self.selector = None
        self._is_syncing = False  # guard to prevent recursive axis callbacks
        self._images_ready = False  # defer axis sync until images initialized

        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.enable_selector()
        # Custom Reset ROI button (bottom-left)
        from matplotlib.widgets import Button, TextBox, RadioButtons
        axbtn = self.fig.add_axes([0.01, 0.02, 0.12, 0.06])
        self.btn_roi = Button(axbtn, 'Reset ROI')
        def reset_roi_click(event):
            self.roi = None
            self.hide_roi_overlay()
            print('[ROI Button] Reset ROI')
            self.update()
        self.btn_roi.on_clicked(reset_roi_click)
        # Preset selection (radio buttons)
        preset_labels = list(WINDOW_PRESETS.keys())
        ax_radio = self.fig.add_axes([0.01, 0.14, 0.09, 0.40])
        self.radio_presets = RadioButtons(ax_radio, preset_labels, active=preset_labels.index(self.preset_name) if self.preset_name in preset_labels else 0)
        def on_preset(label):
            self.apply_new_window(preset=label)
        self.radio_presets.on_clicked(on_preset)
        # Text boxes for WL/WW and apply
        ax_wl = self.fig.add_axes([0.15, 0.02, 0.08, 0.05])
        ax_ww = self.fig.add_axes([0.25, 0.02, 0.08, 0.05])
        self.txt_wl = TextBox(ax_wl, 'WL', initial=str(WINDOW_PRESETS.get(self.preset_name, WINDOW_PRESETS['default'])['center']))
        self.txt_ww = TextBox(ax_ww, 'WW', initial=str(WINDOW_PRESETS.get(self.preset_name, WINDOW_PRESETS['default'])['width']))
        ax_apply = self.fig.add_axes([0.35, 0.02, 0.10, 0.05])
        self.btn_apply = Button(ax_apply, 'Apply WW')
        def apply_manual(event):
            try:
                wl = float(self.txt_wl.text)
                ww = float(self.txt_ww.text)
                self.apply_new_window(center=wl, width=ww)
            except Exception as e:
                print(f"[Apply WW] Invalid WL/WW input: {e}")
        self.btn_apply.on_clicked(apply_manual)
        # Index selection (text field + button)
        ax_idx = self.fig.add_axes([0.47, 0.02, 0.08, 0.05])
        self.txt_idx = TextBox(ax_idx, 'Index', initial=str(self.index))
        ax_idx_go = self.fig.add_axes([0.57, 0.02, 0.06, 0.05])
        self.btn_idx_go = Button(ax_idx_go, 'Go')
        def apply_index(event):
            try:
                val = int(float(self.txt_idx.text))
                D, _, _, _ = self.hr.shape
                val = int(np.clip(val, 0, max(0, D - 1)))
                self.index = val
                self.update()
            except Exception as e:
                print(f"[Apply Index] Invalid index: {e}")
        self.btn_idx_go.on_clicked(apply_index)
        print('[Hint] Navigation: Mouse wheel or arrow keys (Up=superior, Down=inferior, Left=previous, Right=next); Home=inferior, End=superior; Drag on HR to select ROI; press r to reset ROI; change presets or set WL/WW and click Apply')
        # sync to toolbar zoom/pan on all axes
        for ax in [self.ax_hr, self.ax_sr, self.ax_lin, self.ax_bic, self.ax_lr]:
            ax.callbacks.connect('xlim_changed', self.on_axes_limits_change)
            ax.callbacks.connect('ylim_changed', self.on_axes_limits_change)
        self.update()

    def update(self):
        D_hr, _, _, _ = (self.hr_hu.shape if self.hr_hu is not None else self.hr.shape)
        D_lr, _, _, _ = self.lr.shape
        # For lazy SR/lin/bic, use D_hr for bounds
        D_sr = D_hr
        D_lin = D_hr
        D_bic = D_hr

        clamped_idx = int(np.clip(self.index, 0, min(D_hr, D_lr, D_sr, D_lin, D_bic) - 1))
        print(f"[Viewer.update] index={clamped_idx} / D={min(D_hr, D_lr, D_sr, D_lin, D_bic)} (0 = inferior/lowest z, {min(D_hr, D_lr, D_sr, D_lin, D_bic)-1} = superior/highest z)")

        # Use HU as authoritative HR source for display & metrics
        hr_plane, axis_len, _ = extract_slice(self.hr_hu if self.hr_hu is not None else self.hr, clamped_idx)
        lr_plane, _, _ = extract_slice(self.lr, clamped_idx)
        # On-demand compute for SR/lin/bic (cache per-slice tensors in [-1,1])
        def _get_sr_slice(idx):
            if idx not in self._sr_cache:
                x = self.lr[idx:idx+1].to(self.device)
                with torch.no_grad():
                    use_cuda = (self.device.type == 'cuda')
                    with torch.amp.autocast('cuda', enabled=use_cuda):
                        y = self.model(x).detach()
                self._sr_cache[idx] = y.cpu()[0,0]
            return self._sr_cache[idx]
        def _get_lin_slice(idx):
            if idx not in self._lin_cache:
                y = F.interpolate(self.lr[idx:idx+1], scale_factor=(self.scale,self.scale), mode='bilinear', align_corners=False)[0,0]
                self._lin_cache[idx] = y
            return self._lin_cache[idx]
        def _get_bic_slice(idx):
            if idx not in self._bic_cache:
                y = F.interpolate(self.lr[idx:idx+1], scale_factor=(self.scale,self.scale), mode='bicubic', align_corners=False)[0,0]
                self._bic_cache[idx] = y
            return self._bic_cache[idx]
        sr_plane = _get_sr_slice(clamped_idx)
        lin_plane = _get_lin_slice(clamped_idx)
        bic_plane = _get_bic_slice(clamped_idx)
        print(f"[Viewer.update] Slice {clamped_idx}/{axis_len-1} | shapes HR={tuple(hr_plane.shape)} SR={tuple(sr_plane.shape)} LR={tuple(lr_plane.shape)} BIL={None if lin_plane is None else tuple(lin_plane.shape)} BIC={None if bic_plane is not None else tuple(bic_plane.shape)}")
        # Slice identity debug (to verify index alignment)
        if self.slice_meta is not None and 0 <= clamped_idx < len(self.slice_meta):
            meta = self.slice_meta[clamped_idx]
            inst = meta.get('InstanceNumber', None)
            uid = meta.get('SOPInstanceUID', '')
            path = meta.get('path', '')
            print(f"[SliceMeta] idx={clamped_idx} InstanceNumber={inst} SOPInstanceUID={uid} Path={path}")

        # If ROI is set (in HR coords), synchronize axes limits across views
        if self.roi:
            x0, y0, x1, y1 = self.roi
            self.apply_axes_limits(x0, y0, x1, y1)

        # per-slice windowing from HU for display
        # LR is in [-1,1] -> back to HU (global), then window
        lr_hu_slice = m11_to_hu_global(lr_plane.unsqueeze(0), -1000.0, 2000.0)[0]
        sr_hu_slice = m11_to_hu_global(sr_plane.unsqueeze(0), -1000.0, 2000.0)[0]
        lin_hu_slice = m11_to_hu_global(lin_plane.unsqueeze(0), -1000.0, 2000.0)[0]
        bic_hu_slice = m11_to_hu_global(bic_plane.unsqueeze(0), -1000.0, 2000.0)[0]
        img_lr = to_display(hu_to_m11_window(lr_hu_slice, self.curr_wl, self.curr_ww))
        img_sr = to_display(hu_to_m11_window(sr_hu_slice, self.curr_wl, self.curr_ww))
        img_hr = to_display(hu_to_m11_window(hr_plane, self.curr_wl, self.curr_ww))
        img_lin = to_display(hu_to_m11_window(lin_hu_slice, self.curr_wl, self.curr_ww))
        img_bic = to_display(hu_to_m11_window(bic_hu_slice, self.curr_wl, self.curr_ww))
        print(f"[Viewer.update] ranges LR=({img_lr.min():.3f},{img_lr.max():.3f}) SR=({img_sr.min():.3f},{img_sr.max():.3f}) HR=({img_hr.min():.3f},{img_hr.max():.3f})")

        if self.im_lr is None:
            self.im_lr = self.ax_lr.imshow(img_lr, cmap='gray', vmin=0, vmax=1, origin='lower')
        else:
            self.im_lr.set_data(img_lr)
        if self.im_lin is None:
            self.im_lin = self.ax_lin.imshow(img_lin, cmap='gray', vmin=0, vmax=1, origin='lower')
        else:
            self.im_lin.set_data(img_lin)
        if self.im_bic is None:
            self.im_bic = self.ax_bic.imshow(img_bic, cmap='gray', vmin=0, vmax=1, origin='lower')
        else:
            self.im_bic.set_data(img_bic)
        if self.im_sr is None:
            self.im_sr = self.ax_sr.imshow(img_sr, cmap='gray', vmin=0, vmax=1, origin='lower')
        else:
            self.im_sr.set_data(img_sr)
        if self.im_hr is None:
            self.im_hr = self.ax_hr.imshow(img_hr, cmap='gray', vmin=0, vmax=1, origin='lower')
        else:
            self.im_hr.set_data(img_hr)

        # Info panel (always visible)
        info_parts = []
        if self.patient_id:
            info_parts.append(f"PatientID: {self.patient_id}")
        if self.slice_thickness_mm is not None:
            info_parts.append(f"Slice Thickness: {self.slice_thickness_mm:.2f} mm")
        if self.ps_row_mm and self.ps_col_mm:
            info_parts.append(f"PixelSpacing: {self.ps_row_mm:.3f} x {self.ps_col_mm:.3f} mm")
        # Degradation config summary
        try:
            if self.degradation:
                deg_items = [f"Deg: {self.degradation}, "]
                if self.degradation in ('blur','blurnoise'):
                    # Always show sigma range (explicit or auto default based on scale)
                    sigma_auto = False
                    if self.blur_sigma_range is not None:
                        try:
                            s0 = float(self.blur_sigma_range[0]); s1 = float(self.blur_sigma_range[1])
                        except Exception:
                            s0 = self.blur_sigma_range[0]; s1 = self.blur_sigma_range[1]
                    else:
                        # derive default display range from scale (same heuristic as generation)
                        base_sigma = 0.8 if int(self.scale) == 2 else (1.2 if int(self.scale) == 4 else 0.8)
                        jitter = 0.1 if int(self.scale) == 2 else 0.15
                        s0, s1 = max(1e-6, base_sigma - jitter), base_sigma + jitter
                        sigma_auto = True
                    sigma_txt = f"sigma=[{s0:.3f},{s1:.3f}]"
                    if sigma_auto:
                        sigma_txt += " (auto)"
                    deg_items.append(sigma_txt)
                    # Always show kernel (explicit or auto). If auto, compute representative kernel for mid-sigma.
                    if self.blur_kernel is not None:
                        try:
                            kdisp = int(self.blur_kernel)
                        except Exception:
                            kdisp = self.blur_kernel
                        deg_items.append(f"k={kdisp}")
                    else:
                        try:
                            mid_sigma = 0.5 * (s0 + s1)
                            k_auto = _kernel_size_from_sigma(mid_sigma)
                            deg_items.append(f"k={k_auto} (auto)")
                        except Exception:
                            deg_items.append("k=auto")
                if self.degradation == 'blurnoise':
                    if self.noise_sigma_range_norm is not None:
                        try:
                            deg_items.append(f"noise=[{float(self.noise_sigma_range_norm[0]):.4f},{float(self.noise_sigma_range_norm[1]):.4f}]")
                        except Exception:
                            pass
                    if self.dose_factor_range is not None:
                        try:
                            deg_items.append(f"dose=[{float(self.dose_factor_range[0]):.2f},{float(self.dose_factor_range[1]):.2f}]")
                        except Exception:
                            pass
                # Also show effectively used sampled values (if available)
                try:
                    if self.used_sigma is not None:
                        deg_items.append(f"used_sigma={float(self.used_sigma):.4f}")
                except Exception:
                    pass
                try:
                    if self.used_noise_sigma is not None:
                        deg_items.append(f"used_noise_sigma={float(self.used_noise_sigma):.4f}")
                except Exception:
                    pass
                try:
                    if self.used_dose is not None:
                        deg_items.append(f"used_dose={float(self.used_dose):.3f}")
                except Exception:
                    pass
                if self.degradation == 'clean' and self.antialias_clean:
                    deg_items.append("AA")
                if self.degradation_sampling:
                    smp = f"smp={self.degradation_sampling}"
                    if self.deg_seed is not None:
                        smp += f", seed={self.deg_seed}"
                    deg_items.append(smp)
                info_parts.append(', '.join(deg_items))
        except Exception:
            pass
        self.text_info.set_text(' | '.join(info_parts))
        self.text.set_text(f'Index: {clamped_idx}/{axis_len-1} (0=inferior, {axis_len-1}=superior)')

        # If no ROI is active, ensure axes show full images explicitly
        if not self.roi:
            h_lr, w_lr = img_lr.shape
            h_hr, w_hr = img_hr.shape
            self._is_syncing = True
            try:
                self.ax_lr.set_xlim(-0.5, w_lr-0.5); self.ax_lr.set_ylim(h_lr-0.5, -0.5)
                for ax in [self.ax_lin, self.ax_bic, self.ax_sr, self.ax_hr]:
                    ax.set_xlim(-0.5, w_hr-0.5); ax.set_ylim(h_hr-0.5, -0.5)
            finally:
                self._is_syncing = False

        # Determine ROI arrays for metrics (crop if ROI exists) using non-displayed tensors to avoid previous crops
        x0i = y0i = x1i = y1i = None
        H, W = self.hr.shape[-2:]
        # Always show ROI status line; if no ROI, treat as full image
        if self.roi:
            x0, y0, x1, y1 = self.roi
        else:
            x0, y0, x1, y1 = 0, 0, W, H
        x0i, x1i = int(round(max(0, min(x0, W-1)))), int(round(max(0, min(x1, W))))
        y0i, y1i = int(round(max(0, min(y0, H-1)))), int(round(max(0, min(y1, H))))
        # ROI overlay text with px and mm + ROI FOV; also include LR resolution
        w_px = max(0, x1i - x0i)
        h_px = max(0, y1i - y0i)
        if self.ps_row_mm and self.ps_col_mm:
            w_mm = w_px * self.ps_col_mm
            h_mm = h_px * self.ps_row_mm
            fov_w_mm = W * self.ps_col_mm
            fov_h_mm = H * self.ps_row_mm
            roi_text = f"ROI HR: ({x0i},{y0i})-({x1i},{y1i}) px | {w_px}x{h_px} px | ROI FOV {w_mm:.1f}x{h_mm:.1f} mm | Global FOV {fov_w_mm:.1f}x{fov_h_mm:.1f} mm"
        else:
            roi_text = f"ROI HR: ({x0i},{y0i})-({x1i},{y1i}) px | {w_px}x{h_px} px"
        # map to LR coords (may be fractional due to scale; display as-is)
        x0_lr = x0i / float(self.scale)
        y0_lr = y0i / float(self.scale)
        x1_lr = x1i / float(self.scale)
        y1_lr = y1i / float(self.scale)
        lr_h, lr_w = self.lr.shape[-2:]
        roi_text += f" | mapped LR: ({x0_lr:.2f},{y0_lr:.2f})-({x1_lr:.2f},{y1_lr:.2f}) | LR res: {lr_w}x{lr_h} px"
        self.text_roi.set_text(roi_text)

        def crop_roi_full(vol):
            t = vol[clamped_idx, 0]
            if self.roi and x1i is not None and x1i > x0i and y1i > y0i:
                return t[y0i:y1i, x0i:x1i]
            return t

        # Prepare HU planes for metrics (identical ROI across methods)
        def crop_roi_hu(vol):
            t = vol[clamped_idx, 0]
            if self.roi and x1i is not None and x1i > x0i and y1i > y0i:
                return t[y0i:y1i, x0i:x1i]
            return t

        # Use HU slices prepared above (per-slice), then apply ROI cropping uniformly
        def crop_hu2d(x2d):
            if x2d is None:
                return None
            if self.roi and x1i is not None and x1i > x0i and y1i > y0i:
                return x2d[y0i:y1i, x0i:x1i]
            return x2d
        hr_hu = crop_hu2d(hr_plane)
        sr_hu = crop_hu2d(sr_hu_slice)
        lin_hu = crop_hu2d(lin_hu_slice)
        bic_hu = crop_hu2d(bic_hu_slice)
        lr_hu = crop_hu2d(lr_hu_slice)

        # Debug domain logs (reduced):
        # def _dbg(name, x):
        #     if x is None:
        #         return
        #     print(f"[METDBG] {name}: shape={tuple(x.shape)} min={float(x.min()):.3f} max={float(x.max()):.3f} mean={float(x.mean()):.3f}")
        # _dbg("HR(HU)", hr_hu)
        # _dbg("SR(HU)", sr_hu)
        # _dbg("LIN(HU)", lin_hu)
        # _dbg("BIC(HU)", bic_hu)
        # _dbg("LR(HU)", lr_hu)

        if hr_hu is not None and sr_hu is not None:
            d_sr_hr = torch.mean(torch.abs(sr_hu - hr_hu)).item()
            # print(f"[METDBG] mean|SR-HR|={d_sr_hr:.6g}")
            assert d_sr_hr > 1e-6, "SR≅HR → falsches Tensor beim Metrik-Call"
        if hr_hu is not None and bic_hu is not None:
            d_bic_hr = torch.mean(torch.abs(bic_hu - hr_hu)).item()
            # print(f"[METDBG] mean|BIC-HR|={d_bic_hr:.6g}")
            assert d_bic_hr > 1e-6, "Bicubic≅HR → falsches Tensor beim Metrik-Call"
        if hr_hu is not None and lin_hu is not None:
            d_lin_hr = torch.mean(torch.abs(lin_hu - hr_hu)).item()
            # print(f"[METDBG] mean|BIL-HR|={d_lin_hr:.6g}")
            assert d_lin_hr > 1e-4, "Bilinear≅HR → falsches Tensor beim Metrik-Call"

        metrics = {}
        if sr_hu is not None and hr_hu is not None:
            ms = compute_all_metrics(sr_hu, hr_hu, mode='window', wl=self.curr_wl, ww=self.curr_ww, lpips_backbone='alex', device='cuda', return_components=True)
            metrics['SR'] = (ms['MSE'], ms['RMSE'], ms['MAE'], ms['PSNR'], ms['SSIM'], ms['LPIPS'], ms['PI'])
        if lin_hu is not None and hr_hu is not None:
            ml = compute_all_metrics(lin_hu, hr_hu, mode='window', wl=self.curr_wl, ww=self.curr_ww, lpips_backbone='alex', device='cuda', return_components=True)
            metrics['Bilinear'] = (ml['MSE'], ml['RMSE'], ml['MAE'], ml['PSNR'], ml['SSIM'], ml['LPIPS'], ml['PI'])
        if bic_hu is not None and hr_hu is not None:
            mb = compute_all_metrics(bic_hu, hr_hu, mode='window', wl=self.curr_wl, ww=self.curr_ww, lpips_backbone='alex', device='cuda', return_components=True)
            metrics['Bicubic'] = (mb['MSE'], mb['RMSE'], mb['MAE'], mb['PSNR'], mb['SSIM'], mb['LPIPS'], mb['PI'])

        # Clear previous metric texts
        for t in self.metric_texts:
            try:
                t.remove()
            except Exception:
                pass
        self.metric_texts = []

       # Find best per metric (MSE/RMSE/MAE min, PSNR/SSIM max) across all methods present
        names = list(metrics.keys())
        by_metric = {'MSE': {}, 'RMSE': {}, 'MAE': {}, 'PSNR': {}, 'SSIM': {}, 'LPIPS': {}, 'PI': {}}
        for name in names:
            mse_val, rmse_val, mae_val, psnr_val, ssim_val, lpips_val, pi_val = metrics[name]
            by_metric['MSE'][name] = mse_val
            by_metric['RMSE'][name] = rmse_val
            by_metric['MAE'][name] = mae_val  # MAE hinzugefügt
            by_metric['PSNR'][name] = psnr_val
            by_metric['SSIM'][name] = ssim_val
            by_metric['LPIPS'][name] = lpips_val
            by_metric['PI'][name] = pi_val
        print(f"[Viewer.update] metrics: {metrics}")
        best = {
            'MSE': min(by_metric['MSE'], key=by_metric['MSE'].get),
            'RMSE': min(by_metric['RMSE'], key=by_metric['RMSE'].get),
            'MAE': min(by_metric['MAE'], key=by_metric['MAE'].get),
            'PSNR': max(by_metric['PSNR'], key=by_metric['PSNR'].get),
            'SSIM': max(by_metric['SSIM'], key=by_metric['SSIM'].get),
            'LPIPS': min(by_metric['LPIPS'], key=by_metric['LPIPS'].get),
            'PI': min(by_metric['PI'], key=by_metric['PI'].get),
        }

        # Layout metric labels and values in one row; best ones green
        # Headers with left column label
        header_y = 0.965
        def put_header(x, txt):
            self.metric_texts.append(self.fig.text(x, header_y, txt, ha='left', va='top', family='monospace', color='black'))
        put_header(0.02, "Metrics")
        put_header(0.06, "MSE ↓")
        put_header(0.16, "RMSE ↓")
        put_header(0.28, "MAE ↓")
        put_header(0.38, "PSNR ↑")
        put_header(0.50, "SSIM ↑")
        put_header(0.62, "LPIPS ↓")
        put_header(0.74, "PI ↓")

        y0_text = 0.93
        dy = 0.047
        for i, name in enumerate(['SR', 'Bilinear', 'Bicubic']):
            if name not in metrics:
                continue
            mse_val, rmse_val, mae_val, psnr_val, ssim_val, lpips_val, pi_val = metrics[name]
            # single row per method with columns in the specified order
            y = y0_text - i*dy
            def put_val(x, value, is_best, fmt):
                color = 'green' if is_best else 'black'
                self.metric_texts.append(self.fig.text(x, y, fmt.format(value), ha='left', va='top', color=color, family='monospace'))
            self.metric_texts.append(self.fig.text(0.02, y, f"{name}", ha='left', va='top', family='monospace', color='black'))
            put_val(0.06, mse_val, name == best['MSE'], "{:.6f}")
            put_val(0.16, rmse_val, name == best['RMSE'], "{:.6f}")
            put_val(0.28, mae_val, name == best['MAE'], "{:.6f}")
            put_val(0.38, psnr_val, name == best['PSNR'], "{:.2f}")
            put_val(0.50, ssim_val, name == best['SSIM'], "{:.4f}")
            put_val(0.62, lpips_val, name == best['LPIPS'], "{:.4f}")
            put_val(0.74, pi_val, name == best['PI'], "{:.3f}")
        self._images_ready = True
        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        step = 1 if getattr(event, 'step', 0) >= 0 else -1
        if event.button == 'up':
            step = 1
        elif event.button == 'down':
            step = -1
        old_index = self.index
        D, _, _, _ = self.hr.shape
        # clamp index strictly within [0, D-1]
        self.index = int(np.clip(self.index + step, 0, D - 1))
        print(f"[Scroll] Slice {old_index} -> {self.index} (0=inferior, {D-1}=superior)")
        self.update()

    def on_key(self, event):
        if event.key in ['r', 'R']:
            # reset ROI and re-enable selector
            self.roi = None
            self.hide_roi_overlay()
            self.update()
        elif event.key == 'home':
            # Go to inferior (lowest z)
            self.index = 0
            print(f"[Key] Home -> Slice 0 (inferior)")
            self.update()
        elif event.key == 'end':
            # Go to superior (highest z)
            D, _, _, _ = self.hr.shape
            self.index = D - 1
            print(f"[Key] End -> Slice {D-1} (superior)")
            self.update()
        elif event.key in ['left', 'down']:
            # Previous slice: move towards inferior
            if self.index > 0:
                self.index = max(self.index - 1, 0)
                print(f"[Key] Previous -> Slice {self.index}")
                self.update()
        elif event.key in ['right', 'up']:
            # Next slice: move towards superior
            D, _, _, _ = self.hr.shape
            if self.index < D - 1:
                self.index = min(self.index + 1, D - 1)
                print(f"[Key] Next -> Slice {self.index}")
                self.update()

    def enable_selector(self):
        from matplotlib.widgets import RectangleSelector
        # Create selector on HR axes to define ROI in HR coordinates
        if self.selector is not None:
            try:
                self.selector.disconnect_events()
            except Exception:
                pass
            self.selector = None

        def onselect(eclick, erelease):
            x0, y0 = eclick.xdata, eclick.ydata
            x1, y1 = erelease.xdata, erelease.ydata
            if x0 is None or y0 is None or x1 is None or y1 is None:
                return
            # Normalize and clip to integer pixel indices of HR
            H, W = self.hr.shape[-2:]
            x0n, x1n = (x0, x1) if x0 <= x1 else (x1, x0)
            y0n, y1n = (y0, y1) if y0 <= y1 else (y1, y0)
            x0n = max(0, min(int(round(x0n)), W-1))
            x1n = max(0, min(int(round(x1n)), W))
            y0n = max(0, min(int(round(y0n)), H-1))
            y1n = max(0, min(int(round(y1n)), H))
            # enforce minimum size to avoid empty ROI
            if x1n <= x0n + 1 or y1n <= y0n + 1:
                return
            self.roi = (x0n, y0n, x1n, y1n)
            print(f"[RectangleSelector] ROI HR: ({x0n}, {y0n}, {x1n}, {y1n}) | mapped LR: ({x0n/self.scale:.2f}, {y0n/self.scale:.2f}, {x1n/self.scale:.2f}, {y1n/self.scale:.2f})")
            # Sync axes to ROI immediately
            self.hide_roi_overlay()
            self.apply_axes_limits(x0n, y0n, x1n, y1n)
            self.update()

        self.selector = RectangleSelector(
            self.ax_hr, onselect,
            useblit=True, button=[1],  # left mouse drag
            minspanx=5, minspany=5, interactive=True,
            spancoords='pixels'
        )
        self.selector.set_active(True)

    def hide_roi_overlay(self):
        # Remove any red rectangle artifacts from previous selector draws
        try:
            for attr in ['to_draw', 'artists']:
                if hasattr(self.selector, attr):
                    for artist in getattr(self.selector, attr):
                        artist.set_visible(False)
        except Exception:
            pass

    def on_axes_limits_change(self, ax):
        # Sync other axes when any view is zoomed/panned; compute ROI in HR coords
        if self._is_syncing or not self._images_ready:
            return
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        # ignore spurious callbacks with tiny height/width (e.g., during init)
        if (x1 - x0) < 5 or (y1 - y0) < 5:
            return
        # map LR axes to HR coordinates via scale
        if ax is self.ax_lr:
            fx = float(self.scale); fy = float(self.scale)
            xr0, xr1 = x0*fx, x1*fx
            yr0, yr1 = y0*fy, y1*fy
        else:
            xr0, xr1 = x0, x1
            yr0, yr1 = y0, y1
        # store ROI in HR coordinates; normalize to ascending
        xr0, xr1 = (xr0, xr1) if xr0 < xr1 else (xr1, xr0)
        yr0, yr1 = (yr0, yr1) if yr0 < yr1 else (yr1, yr0)
        self.roi = (xr0, yr0, xr1, yr1)
        print(f"[AxesChanged] src={ax.get_title()} xlim=({x0:.1f},{x1:.1f}) ylim=({y0:.1f},{y1:.1f}) -> HR ROI={self.roi}")
        self.apply_axes_limits(xr0, yr0, xr1, yr1)
        self.update()

    def apply_axes_limits(self, x0, y0, x1, y1):
        # Apply HR ROI to SR/LIN/BIC axes; map to LR via scale
        if x1 <= x0 or y1 <= y0:
            return
        try:
            self._is_syncing = True
            # Set HR/SR/LIN/BIC limits directly in HR coords
            for ax in [self.ax_hr, self.ax_sr, self.ax_lin, self.ax_bic]:
                if ax is not None:
                    ax.set_xlim(x0-0.5, x1-0.5)
                    ax.set_ylim(y1-0.5, y0-0.5)
            # LR mapping with scale
            fx = float(self.scale)
            fy = float(self.scale)
            self.ax_lr.set_xlim(x0/fx-0.5, x1/fx-0.5)
            self.ax_lr.set_ylim(y1/fy-0.5, y0/fy-0.5)
            print(f"[ApplyLimits] HR=({x0},{y0})-({x1},{y1}) | LR=({x0/fx:.2f},{y0/fy:.2f})-({x1/fx:.2f},{y1/fy:.2f})")
        finally:
            self._is_syncing = False

    def apply_new_window(self, preset=None, center=None, width=None):
        # Bestimme WL/WW
        if preset is not None:
            self.preset_name = preset
            cfg = WINDOW_PRESETS.get(self.preset_name, WINDOW_PRESETS['default'])
            wl = float(cfg['center'])
            ww = float(cfg['width'])
            # Textboxen aktualisieren
            try:
                self.txt_wl.set_val(str(wl))
                self.txt_ww.set_val(str(ww))
            except Exception:
                pass
        else:
            wl = float(center)
            ww = float(width)

        # Refenstern der Anzeigevolumes aus den bereits gespeicherten HU-Volumes
        try:
            if self.hr_hu is not None:
                self.hr = hu_to_m11_window(self.hr_hu, wl, ww)
            if self.sr_hu is not None:
                self.sr = hu_to_m11_window(self.sr_hu, wl, ww)
            if self.lin_hu is not None:
                self.lin = hu_to_m11_window(self.lin_hu, wl, ww)
            if self.bic_hu is not None:
                self.bic = hu_to_m11_window(self.bic_hu, wl, ww)
            if self.lr_hu is not None:                     # <-- vereinfacht
                self.lr = hu_to_m11_window(self.lr_hu, wl, ww)
        except ValueError as e:
            print(f"[Apply WW] Invalid WL/WW input: {e}")
            return

        # store current windowing for metrics
        self.curr_wl, self.curr_ww = wl, ww

        # Konsolen-Log der WL/WW-Effekte (neue Min/Max/Mittelwerte im aktuellen Slice)
        try:
            D = self.hr_hu.shape[0]
            idx = int(np.clip(self.index, 0, D-1))
            def stats(x):
                return float(x.min()), float(x.max()), float(x.mean())
            hr_min, hr_max, hr_mean = stats(self.hr_hu[idx, 0])
            sr_min, sr_max, sr_mean = stats(self.sr_hu[idx, 0]) if self.sr_hu is not None else (float('nan'),)*3
            print(f"[WL/WW] preset={self.preset_name} WL={wl:.1f} WW={ww:.1f} | HR(HU) min/max/mean={hr_min:.1f}/{hr_max:.1f}/{hr_mean:.1f} | SR(HU) min/max/mean={sr_min:.1f}/{sr_max:.1f}/{sr_mean:.1f}")
        except Exception:
            pass

        # Short debug print of display ranges after applying WL/WW
        try:
            D_disp = self.hr.shape[0]
            idx_disp = int(np.clip(self.index, 0, D_disp-1))
            def disp_rng(tensor_vol):
                if tensor_vol is None:
                    return (float('nan'), float('nan'))
                img = to_display(tensor_vol[idx_disp, 0])
                return float(img.min()), float(img.max())
            lr_min_d, lr_max_d = disp_rng(self.lr)
            sr_min_d, sr_max_d = disp_rng(self.sr)
            hr_min_d, hr_max_d = disp_rng(self.hr)
            print(f"[Window] Applied WL/WW=({wl:.1f},{ww:.1f}); ranges LR=({lr_min_d:.3f},{lr_max_d:.3f}) SR=({sr_min_d:.3f},{sr_max_d:.3f}) HR=({hr_min_d:.3f},{hr_max_d:.3f})")
        except Exception:
            pass

        # ROI & Index bleiben erhalten; einfach neu zeichnen
        self.update()



    def compute_metrics(self, sr_plane_t, hr_plane_t):
        # Deprecated: not used; kept for compatibility
        m = compute_all_metrics(sr_plane_t, hr_plane_t, mode='global', hu_clip=(-1000.0, 2000.0), lpips_backbone='alex', device='cpu', return_components=True)
        return (m['MSE'], m['RMSE'], m['MAE'], m['PSNR'], m['SSIM'], m['LPIPS'], m['PI'])


def main():
    import argparse
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import numpy as np

    from rrdb_ct_model import RRDBNet_CT
    from window_presets import WINDOW_PRESETS

    # ---------- CLI ----------
    parser = argparse.ArgumentParser(description='Visualize LR vs SR vs HR CT slices with mouse-wheel scrolling')
    parser.add_argument('--dicom_folder', type=str, required=True, help='Root folder containing DICOM series')
    parser.add_argument('--preset', type=str, default='soft_tissue', help='Window preset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--scale', type=int, default=2, help='Upsampling scale (must match model)')
    parser.add_argument('--sr_batch', type=int, default=20, help='Batch size for batched SR inference (speed up GUI)')
    # Degradation flags (default blurnoise)
    parser.add_argument('--degradation', type=str, default='blurnoise', choices=['clean', 'blur', 'blurnoise'], help='Degradation pipeline for LR generation (default: blurnoise)')
    parser.add_argument('--blur_sigma_range', type=float, nargs=2, default=None, help='Range [lo hi] of Gaussian blur sigma; if None, defaults by scale')
    parser.add_argument('--blur_kernel', type=int, default=None, help='Explicit odd kernel size; if None, derived from sigma')
    parser.add_argument('--noise_sigma_range_norm', type=float, nargs=2, default=[0.001, 0.003], help='Gaussian noise sigma range on normalized [-1,1] image')
    parser.add_argument('--dose_factor_range', type=float, nargs=2, default=[0.25, 0.5], help='Dose factor range; noise scales ~ 1/sqrt(dose)')
    parser.add_argument('--antialias_clean', action='store_true', help='Use antialias in clean downsample')
    parser.add_argument('--degradation_sampling', type=str, default='volume', choices=['volume','slice','det-slice'], help='Degradation sampling mode (volume|slice|det-slice) [viewer applies uniformly per volume]')
    parser.add_argument('--deg_seed_mode', type=str, default='per_patient', choices=['global','per_patient'], help='How to derive degradation seed (global fixed vs per_patient hashed by path)')
    parser.add_argument('--deg_seed', type=int, default=42, help='Base seed for degradation sampling in viewer')
    parser.add_argument('--eval_summary', type=str, default=None, help='Optional path to evaluation summary.json to reuse per-patient sampled degradation params')
    args = parser.parse_args()

    # ---------- model ----------
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    model = RRDBNet_CT(scale=args.scale).to(device)
    state = torch.load(args.model_path, map_location=device)
    if isinstance(state, dict) and 'model' in state and all(k in state for k in ['epoch', 'model']):
        print("[Vis] Detected checkpoint dict; loading weights from 'model' key")
        state = state['model']
    model.load_state_dict(state)
    model.eval()




    # ---------- Daten laden (HU + Meta) ----------
    hr_hu_vol, slice_meta = load_series_hu(args.dicom_folder)  # HU via pydicom.apply_modality_lut
    (row_mm, col_mm), slice_thickness, patient_id = read_series_metadata(args.dicom_folder)

    # Center-Crop HU auf Scale-Multiples (vor jeder Normalisierung!)
    D, C, H, W = hr_hu_vol.shape
    target_H = (H // args.scale) * args.scale
    target_W = (W // args.scale) * args.scale
    if target_H > 0 and target_W > 0 and (target_H != H or target_W != W):
        y0 = (H - target_H) // 2
        x0 = (W - target_W) // 2
        hr_hu_vol = hr_hu_vol[:, :, y0:y0 + target_H, x0:x0 + target_W]
        print(f"[Vis] Center-cropped HR from ({H},{W}) -> ({target_H},{target_W}) to match scale={args.scale}")

    # ---------- Global normalisieren (für Degradation/Modell) ----------
    lo, hi = -1000.0, 2000.0
    hr_norm_vol = hu_to_m11_global(hr_hu_vol, lo, hi)  # [-1,1]

    print(f"[Vis] Degradation='{args.degradation}' | blur_sigma_range={args.blur_sigma_range} | blur_kernel={args.blur_kernel} | noise_sigma_range_norm={args.noise_sigma_range_norm} | dose_factor_range={args.dose_factor_range}")

    # If eval_summary is provided, try to load per-patient sampled params and enforce them
    enforced_params = None
    if args.eval_summary is not None:
        try:
            import json
            with open(args.eval_summary, 'r') as f:
                summ = json.load(f)
            # patient id was parsed from dicoms above
            p2s = summ.get('patient_to_degradation_sampled') or {}
            # Try multiple candidate keys to match evaluation patient identifiers
            base_folder = os.path.basename(args.dicom_folder.rstrip('/\\'))
            candidates = []
            # direct PatientID as in DICOM
            if isinstance(patient_id, str) and len(patient_id) > 0:
                candidates.append(patient_id)
                # common suffix used in preprocessed folders
                if not patient_id.endswith('pp'):
                    candidates.append(patient_id + 'pp')
                else:
                    candidates.append(patient_id[:-2])
            # base folder name (e.g., 14655pp)
            if isinstance(base_folder, str) and len(base_folder) > 0:
                candidates.append(base_folder)
                if base_folder.endswith('pp'):
                    candidates.append(base_folder[:-2])
                else:
                    candidates.append(base_folder + 'pp')
            # deduplicate preserving order
            seen = set()
            candidates = [c for c in candidates if (c not in seen and not seen.add(c))]
            matched_key = None
            for key in candidates:
                if key in p2s:
                    matched_key = key
                    enforced_params = p2s[key]
                    break
            if matched_key is None:
                print(f"[Vis] No match in eval summary for candidates={candidates}; falling back to sampling.")
            else:
                print(f"[Vis] Using sampled degradation from eval summary for key={matched_key}: {enforced_params}")
            if enforced_params:
                print(f"[Vis] Using sampled degradation from eval summary for patient={patient_id}: {enforced_params}")
                # Enforce as point intervals
                try:
                    if 'sigma' in enforced_params:
                        args.blur_sigma_range = [float(enforced_params['sigma']), float(enforced_params['sigma'])]
                    if 'kernel' in enforced_params and (args.blur_kernel is None):
                        args.blur_kernel = int(enforced_params['kernel'])
                    if 'noise_sigma' in enforced_params:
                        args.noise_sigma_range_norm = [float(enforced_params['noise_sigma']), float(enforced_params['noise_sigma'])]
                    if 'dose' in enforced_params:
                        args.dose_factor_range = [float(enforced_params['dose']), float(enforced_params['dose'])]
                except Exception:
                    pass
        except Exception as e:
            print(f"[Vis] Failed to load eval summary: {e}")

    used_seed = int(args.deg_seed) if str(args.deg_seed_mode) == 'global' else fixed_seed_for_path(args.dicom_folder, int(args.deg_seed))
    print(f"[Vis] Degradation sampling='{args.degradation_sampling}' (viewer uses per-volume) | deg_seed_mode={args.deg_seed_mode} | base_seed={args.deg_seed} | used_seed={used_seed}")
    # RNG for degradation sampling (per-volume)
    rng_vis = np.random.default_rng(int(used_seed))
    lr_vol, used_deg = build_lr_volume_from_hr(
        hr_norm_vol, scale=args.scale,
        degradation=args.degradation,
        blur_sigma_range=args.blur_sigma_range,
        blur_kernel=args.blur_kernel,
        noise_sigma_range_norm=args.noise_sigma_range_norm,
        dose_factor_range=args.dose_factor_range,
        antialias_clean=args.antialias_clean,
        rng=rng_vis
    )

    # ---- Log effective degradation parameters (range + drawn values), matching evaluator style ----
    try:
        if args.degradation in ('blur', 'blurnoise'):
            if args.blur_sigma_range is None:
                base_sigma = 0.8 if args.scale == 2 else (1.2 if args.scale == 4 else 0.8)
                jitter = 0.1 if args.scale == 2 else 0.15
                eff_lo, eff_hi = max(1e-6, base_sigma - jitter), base_sigma + jitter
                sigma_note = 'auto'
            else:
                eff_lo, eff_hi = float(args.blur_sigma_range[0]), float(args.blur_sigma_range[1])
                sigma_note = 'explicit'
            k_used = used_deg.get('blur_kernel_k')
            k_note = 'explicit' if (args.blur_kernel is not None) else 'auto'
            s_used = used_deg.get('blur_sigma')
            n_used = used_deg.get('noise_sigma')
            d_used = used_deg.get('dose')
            print(f"[Vis-Degrad] sigma_range=[{eff_lo:.4f},{eff_hi:.4f}] ({sigma_note}) | used_sigma={s_used if s_used is not None else 'NA'} | k={k_used if k_used is not None else 'NA'} ({k_note})")
            if args.degradation == 'blurnoise':
                print(f"[Vis-Degrad] noise_sigma_range={args.noise_sigma_range_norm} | dose_factor_range={args.dose_factor_range} | used_noise_sigma={n_used if n_used is not None else 'NA'} | used_dose={d_used if d_used is not None else 'NA'}")
        else:
            print(f"[Vis-Degrad] clean | antialias={args.antialias_clean}")
    except Exception:
        pass

    # ---------- SR/Interpolationen: Lazy-on-demand in Viewer (keine upfront Berechnung) ----------

    # ---------- Für Metriken: (Lazy) – Volumenweite Konvertierung entfällt; per Slice in Viewer ----------

    # ---------- Viewer starten (Lazy Fensterung & Lazy SR/Interpolationen) ----------
    preset_cfg = WINDOW_PRESETS.get(args.preset, WINDOW_PRESETS['default'])
    wl = float(preset_cfg['center']); ww = float(preset_cfg['width'])
    viewer = ViewerLRSRHR(
        lr_vol, None, hr_norm_vol,
        scale=args.scale,
        lin_volume=None,
        bic_volume=None,
        dicom_folder=args.dicom_folder,
        preset_name=args.preset,
        model=model,
        device=device,
        pixel_spacing_mm=(row_mm, col_mm) if (row_mm is not None and col_mm is not None) else None,
        slice_thickness_mm=slice_thickness,
        patient_id=patient_id,
        hr_hu_volume=hr_hu_vol,
        sr_hu_volume=None,
        lin_hu_volume=None,
        bic_hu_volume=None,
        lr_hu_volume=None,
        slice_meta=slice_meta,
        degradation=args.degradation,
        blur_sigma_range=args.blur_sigma_range,
        blur_kernel=args.blur_kernel,
        noise_sigma_range_norm=args.noise_sigma_range_norm,
        dose_factor_range=args.dose_factor_range,
        antialias_clean=args.antialias_clean,
        degradation_sampling=args.degradation_sampling,
        deg_seed=used_seed,
    )
    # pass effective used degradation values to the viewer for display only
    try:
        viewer.used_sigma = used_deg.get('blur_sigma')
        viewer.used_noise_sigma = used_deg.get('noise_sigma')
        viewer.used_dose = used_deg.get('dose')
        # refresh info panel to include used_* values immediately
        try:
            viewer.update()
        except Exception:
            pass
    except Exception:
        pass

    print('Navigation: Mouse wheel or arrow keys to navigate slices')
    print('Keyboard shortcuts: Home (inferior), End (superior), Arrow keys (previous/next)')
    print('Indexing: 0 = inferior (lowest z), highest index = superior (highest z)')
    plt.show()



if __name__ == '__main__':
    main() 