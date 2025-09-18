import os
import hashlib
import torch
import pydicom
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from typing import Literal
from pydicom.pixel_data_handlers.util import apply_modality_lut
from downsample_tensor_volume import downsample_tensor


#Loading whole slices is usually not worth it and fills the memory unnecessarily, so smaller random patches
def random_aligned_crop(hr_tensor, *, hr_patch: int, scale: int, seed: int):
    """
    Deterministic HR-only random crop aligned to scale multiples, matching the logic in __getitem__.
    - hr_tensor: [1,H,W]
    - Returns: cropped HR tensor [1, hr_patch, hr_patch] or original if too small.
    - Uses the provided seed to ensure deterministic behavior per epoch/slice.
    """
    _, H, W = hr_tensor.shape
    if hr_patch is None:
        return hr_tensor
    if H < hr_patch or W < hr_patch:
        return hr_tensor
    max_y = H - hr_patch
    max_x = W - hr_patch
    rng = np.random.default_rng(int(seed))
    y0 = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
    x0 = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
    hr_crop = hr_tensor[:, y0:y0+hr_patch, x0:x0+hr_patch]
    return hr_crop




def is_ct_image_dicom(path):
    """
    Returns True only for CT image storage objects. Skips DICOM SEG and any non-CT modalities.
    Uses header-only read for speed and robustness.
    """
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        modality = getattr(ds, 'Modality', '')
        if modality != 'CT':
            return False
        sop_class = str(getattr(ds, 'SOPClassUID', ''))
        allowed_sops = {
            '1.2.840.10008.5.1.4.1.1.2',    # CT Image Storage
            '1.2.840.10008.5.1.4.1.1.2.1',  # Enhanced CT Image Storage
        }
        # If SOPClassUID missing but modality CT, still accept to be lenient
        return sop_class in allowed_sops or sop_class == ''
    except Exception:
        return False


def find_dicom_files_recursively(base_folder):
    dicom_files = []
    ignore_dirs = {'ctkDICOM-Database', '.git', '__pycache__'}
    print(f"[CT-Loader] Scanning DICOMs under: {base_folder}")
    for root, dirs, files in os.walk(base_folder):
        # prune ignored directories in-place to avoid descending into them
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for f in files:
            if f.lower().endswith('.dcm'):
                path = os.path.join(root, f)
                if is_ct_image_dicom(path):
                    dicom_files.append(path)
    print(f"[CT-Loader] Found {len(dicom_files)} CT image files")
    return sorted(dicom_files)

def load_dicom_as_tensor(path, hu_clip=(-1000, 2000)):
    """
    Load a DICOM slice as a normalized tensor [1,H,W].
    - Applies Modality LUT (RescaleSlope/Intercept) to obtain HU when present.
    - Always applies global HU clip (default [-1000,2000]) and scales to [-1,1].
    """
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array
    try:
        hu = apply_modality_lut(arr, ds).astype(np.float32)
    except Exception:
        hu = arr.astype(np.float32)

    # Global normalization to [-1,1] with HU clip
    lo, hi = hu_clip
    img = np.clip(hu, lo, hi)
    img = (img - lo) / (hi - lo)  # [0,1]
    img = img * 2 - 1             # [-1,1]
    img = img.astype(np.float32)

    tensor = torch.tensor(img).unsqueeze(0)  # [1, H, W]
    return tensor




def _gaussian_kernel_2d(sigma: float, kernel_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # ensure odd kernel size
    k = int(kernel_size)
    if k % 2 == 0:
        k = k + 1
    half = (k - 1) // 2
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    gauss_1d = torch.exp(-(x ** 2) / (2 * sigma * sigma))
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel_2d = torch.outer(gauss_1d, gauss_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.view(1, 1, k, k)


def gaussian_blur_2d(tensor_1chw: torch.Tensor, sigma: float, kernel_size: int) -> torch.Tensor:
    # tensor_1chw: [1,H,W]
    device = tensor_1chw.device
    dtype = tensor_1chw.dtype
    kernel = _gaussian_kernel_2d(max(1e-6, float(sigma)), kernel_size, device, dtype)
    # pad reflect to preserve size
    k = kernel.shape[-1]
    pad = (k // 2, k // 2, k // 2, k // 2)
    x = tensor_1chw.unsqueeze(0)  # [1,1,H,W]
    x = F.pad(x, pad, mode='reflect')
    out = F.conv2d(x, kernel)
    return out.squeeze(0)  # [1,H,W]


def _compute_kernel_size_from_sigma(sigma: float) -> int:
    # common heuristic: k ~ 6*sigma rounded to nearest odd
    k = int(max(3, round(6.0 * float(sigma))))
    if k % 2 == 0:
        k += 1
    return k

class CT_Dataset_SR(Dataset):
    def __init__(self, dicom_folder, scale_factor=2, max_slices=None,
                 do_random_crop=True, hr_patch=192, hu_clip=(-1000, 2000),
                 degradation='blurnoise', blur_sigma_range=None, blur_kernel=None,
                 noise_sigma_range_norm=(0.001, 0.003), dose_factor_range=(0.25, 0.5), antialias_clean=True,
                 reverse_order=True,
                 degradation_sampling: Literal['volume','slice','det-slice'] = 'volume', deg_seed: int = 42):
        self.paths = find_dicom_files_recursively(dicom_folder)
        self.patient_id = os.path.basename(os.path.normpath(dicom_folder))
        if reverse_order:
            # align ordering with viewer: last loaded slice first (index 0)
            self.paths = list(reversed(self.paths))
        if max_slices:
            self.paths = self.paths[:max_slices]
        self.scale = scale_factor
        self.do_random_crop = do_random_crop
        self.hr_patch = hr_patch
        self.hu_clip = hu_clip
        # degradation settings
        self.degradation = degradation  # 'clean' | 'blur' | 'blurnoise'
        self.degradation_sampling = degradation_sampling
        self.deg_seed = int(deg_seed)
        # default sigma ranges based on scale if not provided
        if blur_sigma_range is None:
            base_sigma = 0.8 if self.scale == 2 else (1.2 if self.scale == 4 else 0.8)
            jitter = 0.1 if self.scale == 2 else 0.15
            self.blur_sigma_range = (max(1e-6, base_sigma - jitter), base_sigma + jitter)
        else:
            self.blur_sigma_range = tuple(blur_sigma_range)
        self.blur_kernel = blur_kernel  # if None, derive per-sample from sigma
        self.noise_sigma_range_norm = tuple(noise_sigma_range_norm)
        self.dose_factor_range = tuple(dose_factor_range)
        self.antialias_clean = bool(antialias_clean)
        # per-volume deterministic degradation params
        self._deg_params = None
        self._deg_logged = False
        # track current epoch for deterministic per-slice sampling (crops/noise)
        self._current_epoch = 0
        if self.degradation_sampling == 'volume' and self.degradation in ('blur','blurnoise'):
            # initial sample for epoch 0 based on base seed + patient
            self.resample_volume_params(epoch_seed=0)
        print(f"[CT-Loader] Dataset ready: {len(self.paths)} slices | scale={self.scale} | norm=global_HU_clip={self.hu_clip} | random_crop={self.do_random_crop}")
        if self.degradation == 'clean':
            # In clean mode, only downsampling is used; blur/noise settings are not applied
            print(f"[CT-Loader] Degradation='clean' | downsample only (antialias={self.antialias_clean}) | blur/noise parameters unused")
            print(f"[CT-Loader] Degradation sampling: n/a for 'clean'")
        else:
            print(f"[CT-Loader] Degradation='{self.degradation}' | blur_sigma_range={self.blur_sigma_range} | blur_kernel={self.blur_kernel} | noise_sigma_range_norm={self.noise_sigma_range_norm} | dose_factor_range={self.dose_factor_range}")
            print(f"[CT-Loader] Degradation sampling mode='{self.degradation_sampling}' | deg_seed={self.deg_seed}")

    def _seed_for_epoch(self, epoch_seed: int) -> int:
        key = f"{int(self.deg_seed)}|{self.patient_id}|{int(epoch_seed)}"
        h = hashlib.sha256(key.encode('utf-8')).hexdigest()
        return int(h[:8], 16)  # 32-bit seed

    def _seed_for_item(self, epoch_seed: int, idx: int, kind: str) -> int:
        """Deterministic per-slice seed derived from base deg_seed, patient id (lowercased), epoch, slice idx and a kind tag.
        kind in {"crop","noise"}.
        """
        pid = str(self.patient_id).lower()
        key = f"{int(self.deg_seed)}|{pid}|{int(epoch_seed)}|{int(idx)}|{str(kind)}"
        h = hashlib.sha256(key.encode('utf-8')).hexdigest()
        return int(h[:8], 16)

    def resample_volume_params(self, epoch_seed: int) -> None:
        """Resample volume-wise degradation parameters deterministically per epoch.
        Uses base deg_seed + patient_id + epoch_seed to draw new (sigma, kernel, noise, dose).
        Effective only when degradation_sampling=='volume' and degradation in ('blur','blurnoise').
        """
        # remember current epoch for per-slice deterministic sampling (crops/noise)
        self._current_epoch = int(epoch_seed)
        if not (self.degradation_sampling == 'volume' and self.degradation in ('blur', 'blurnoise')):
            return
        rng = np.random.default_rng(self._seed_for_epoch(epoch_seed))
        # blur sigma and kernel
        sig_lo, sig_hi = self.blur_sigma_range
        blur_sigma = float(rng.uniform(sig_lo, sig_hi))
        if self.blur_kernel is not None:
            blur_k = int(self.blur_kernel)
            if blur_k % 2 == 0:
                blur_k += 1
        else:
            k = int(max(3, round(6.0 * float(blur_sigma))))
            blur_k = k if (k % 2 == 1) else (k + 1)
        # noise/dose if applicable
        if self.degradation == 'blurnoise':
            n_lo, n_hi = self.noise_sigma_range_norm
            noise_sigma = float(rng.uniform(n_lo, n_hi))
            d_lo, d_hi = self.dose_factor_range
            dose = float(rng.uniform(min(d_lo, d_hi), max(d_lo, d_hi)))
            noise_eff = noise_sigma / max(1e-6, dose) ** 0.5
        else:
            noise_sigma = 0.0
            dose = 1.0
            noise_eff = 0.0
        self._deg_params = {
            'sigma': float(blur_sigma),
            'kernel': int(blur_k),
            'noise_sigma': float(noise_sigma),
            'dose': float(dose),
            'noise_eff': float(noise_eff),
        }
        # public alias used by evaluator
        self.deg_params = dict(self._deg_params)
        # ensure logging happens again on next __getitem__
        self._deg_logged = False
        print(f"[Deg-Resample] patient={self.patient_id} epoch_seed={epoch_seed} sigma={blur_sigma:.4f} k={blur_k} noise_sigma={noise_sigma:.5f} dose={dose:.3f} noise_eff={noise_eff:.5f}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        hr_full = load_dicom_as_tensor(self.paths[idx], hu_clip=self.hu_clip)   # [1, H, W]
        # choose HR region
        if self.do_random_crop and self.hr_patch is not None:
            seed_crop = self._seed_for_item(getattr(self, "_current_epoch", 0), idx, "crop")
            hr = random_aligned_crop(hr_full, hr_patch=self.hr_patch, scale=self.scale, seed=seed_crop)
        else:
            # center-crop to multiples of scale for full-slice evaluation
            hr = hr_full
            _, H, W = hr.shape
            target_H = (H // self.scale) * self.scale
            target_W = (W // self.scale) * self.scale
            if target_H > 0 and target_W > 0 and (target_H != H or target_W != W):
                y0 = (H - target_H) // 2
                x0 = (W - target_W) // 2
                hr = hr[:, y0:y0+target_H, x0:x0+target_W]

        # sample jitter parameters per item/volume/deterministic slice
        blur_sigma_used = None
        noise_sigma_used = None
        dose_used = None
        if self.degradation in ('blur', 'blurnoise'):
            if self.degradation_sampling == 'volume' and self._deg_params is not None:
                blur_sigma_used = float(self._deg_params.get('sigma'))
                k = int(self._deg_params.get('kernel'))
            elif self.degradation_sampling == 'det-slice':
                rng = np.random.default_rng(self.deg_seed + int(idx))
                sig_lo, sig_hi = self.blur_sigma_range
                blur_sigma_used = float(rng.uniform(sig_lo, sig_hi))
                k = self.blur_kernel if self.blur_kernel is not None else _compute_kernel_size_from_sigma(blur_sigma_used)
            else:  # 'slice'
                rng = np.random.default_rng()
                sig_lo, sig_hi = self.blur_sigma_range
                blur_sigma_used = float(rng.uniform(sig_lo, sig_hi))
                k = self.blur_kernel if self.blur_kernel is not None else _compute_kernel_size_from_sigma(blur_sigma_used)
            hr_for_lr = gaussian_blur_2d(hr, sigma=blur_sigma_used, kernel_size=k)
        else:
            hr_for_lr = hr

        # downsample
        if self.degradation == 'clean':
            lr = downsample_tensor(hr_for_lr, self.scale, antialias=self.antialias_clean)
        else:
            # disable antialias to respect explicit blur kernel
            lr = downsample_tensor(hr_for_lr, self.scale, antialias=False)

        # optional noise on LR
        if self.degradation == 'blurnoise':
            if self.degradation_sampling == 'volume' and self._deg_params is not None:
                noise_sigma_used = float(self._deg_params.get('noise_sigma'))
                dose_used = float(self._deg_params.get('dose'))
            elif self.degradation_sampling == 'det-slice':
                rng = np.random.default_rng(self.deg_seed + int(idx))
                n_lo, n_hi = self.noise_sigma_range_norm
                noise_sigma_used = float(rng.uniform(n_lo, n_hi))
                d_lo, d_hi = self.dose_factor_range
                dose_used = float(rng.uniform(min(d_lo, d_hi), max(d_lo, d_hi)))
            else:  # 'slice'
                rng = np.random.default_rng()
                n_lo, n_hi = self.noise_sigma_range_norm
                noise_sigma_used = float(rng.uniform(n_lo, n_hi))
                d_lo, d_hi = self.dose_factor_range
                dose_used = float(rng.uniform(min(d_lo, d_hi), max(d_lo, d_hi)))
            noise_eff = float(noise_sigma_used) / max(1e-6, float(dose_used)) ** 0.5
            # Use numpy RNG for noise, then convert to torch tensor on same device/dtype
            # deterministic noise field per slice and epoch
            seed_noise = self._seed_for_item(getattr(self, "_current_epoch", 0), idx, "noise")
            rng_noise = np.random.default_rng(seed_noise)
            noise_np = rng_noise.normal(loc=0.0, scale=noise_eff, size=lr.shape,)
            noise_t = torch.as_tensor(noise_np, device=lr.device, dtype=lr.dtype)
            lr = lr + noise_t
            lr = torch.clamp(lr, -1.0, 1.0) # clamp to [-1,1]

        # one-time log of actual parameters used
        if not self._deg_logged:
            # print(f"[Deg] mode={self.degradation_sampling} blur_sigma={blur_sigma_used if blur_sigma_used is not None else 'NA'} noise_sigma={noise_sigma_used if noise_sigma_used is not None else 'NA'} dose={dose_used if dose_used is not None else 'NA'}")
            # Note: This Debug-Print shows the effectively used degradation-parameter
            # (σ, noise, dose) an. Useful for verification and debugging, but too much console logs when working with big datasets
            self._deg_logged = True

        return lr, hr