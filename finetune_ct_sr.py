"""
Finetune RRDBNet (1->1 channel) on CT data with ESRGAN-style objectives.

Defaults and rationale:
- Pixel loss (L1): lambda_pix = 1.0
- Perceptual loss (VGG19 conv5_4 pre-ReLU): lambda_perc = 0.08
- Adversarial loss (Relativistic average GAN, RaGAN): lambda_gan = 0.003
- Optimizer: Adam(lr=1e-4, betas=(0.9, 0.999), weight_decay=0)
- Scheduler: MultiStepLR with milestones at 60% and 80% of total epochs (gamma=0.5)
- AMP: torch.cuda.amp with GradScaler
- EMA for generator weights with decay=0.999 (EMA used for validation/checkpointing)
- Gradient clipping for both G and D with max_norm=1.0
- Early stopping monitors Total_NoGAN by default (MAE/PSNR optional)

Conservative weights (lambda_perc=0.08, lambda_gan=0.003) are chosen for medical CT
to reduce hallucinations while still providing sharper textures than pure L1 training.
This prioritizes metric-faithful reconstructions (L1/PSNR/SSIM) over aggressively
hallucinated details.
"""

import os
import sys
import time
import csv
from datetime import datetime
import json
import argparse
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch import amp as torch_amp
from torchvision import models
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt

from rrdb_ct_model import RRDBNet_CT
from ct_dataset_loader import CT_Dataset_SR
from evaluate_ct_model import split_patients, get_patient_dirs
from seed_utils import fixed_seed_for_path
from ct_sr_evaluation import evaluate_metrics
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


class _Tee:
    def __init__(self, log_path: str):
        self._stdout = sys.stdout
        self._fh = open(log_path, 'a', encoding='utf-8', buffering=1)
    def write(self, data: str) -> None:
        self._stdout.write(data)
        self._fh.write(data)
    def flush(self) -> None:
        self._stdout.flush()
        self._fh.flush()



# -----------------------------
# Utility: EMA (Exponential Moving Average) hält „geglättete“ Schattenkopie der Generator‑Gewichte, ändert sich langsamer als die Live‑Gewichte
# -----------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema_model = type(model)() if hasattr(model, '__class__') else None
        # build a copy with same architecture - Schattenmodell
        self.ema_model = RRDBNet_CT(in_nc=1, out_nc=1, nf=64, nb=23, gc=32, scale=model.scale) #Modell wird hier nicht trainiert, sondern dient nur zum Validieren/Speichern der Gewichte
        self.ema_model.load_state_dict(model.state_dict(), strict=True)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.ema_model.state_dict().items():
            if k in msd:
                self.ema_model.state_dict()[k].copy_(v.detach().mul(self.decay).add(msd[k].detach(), alpha=1.0 - self.decay))


# -----------------------------
# Discriminator: PatchGAN with SpectralNorm
# -----------------------------
def spectral_norm(module: nn.Module) -> nn.Module:
    return nn.utils.spectral_norm(module)


class PatchDiscriminatorSN(nn.Module):
    """Patch-based discriminator for 1-channel images. Outputs a score map (logits)."""
    def __init__(self, in_nc: int = 1):
        super().__init__()
        nf = 64
        layers = [
            # Layer 1: 1 Channel (HR-Image) → 64 Channels, H×W remains the same
            spectral_norm(nn.Conv2d(in_nc, nf, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 64 Channels → 64 Channels, H×W halved (stride=2)
            spectral_norm(nn.Conv2d(nf, nf, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 64 Channels → 128 Channels, H×W halved
            spectral_norm(nn.Conv2d(nf, nf * 2, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 128 Channels → 128 Channels, H×W halved (stride=2)
            spectral_norm(nn.Conv2d(nf * 2, nf * 2, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 5: 128 Channels → 256 Channels, H×W halved
            spectral_norm(nn.Conv2d(nf * 2, nf * 4, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 6: 256 Channels → 256 Channels, H×W halved (stride=2)
            spectral_norm(nn.Conv2d(nf * 4, nf * 4, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 7: 256 Channels → 512 Channels, H×W halved
            spectral_norm(nn.Conv2d(nf * 4, nf * 8, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 8: 512 Channels → 512 Channels, H×W halved (stride=2)
            spectral_norm(nn.Conv2d(nf * 8, nf * 8, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 9: 512 Channels → 1 Channel (Logits), H×W stays halved
            spectral_norm(nn.Conv2d(nf * 8, 1, 3, 1, 1))  # Output: Logits-Matrix
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Perceptual Loss (VGG19 features before ReLU)
# -----------------------------
class VGG19FeatureExtractor(nn.Module):
    def __init__(self, layer: str = 'conv5_4'):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        # conv indices in torchvision VGG19 features
        conv_indices = {
            'conv1_1': 0, 'conv1_2': 2,
            'conv2_1': 5, 'conv2_2': 7,
            'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14, 'conv3_4': 16,
            'conv4_1': 19, 'conv4_2': 21, 'conv4_3': 23, 'conv4_4': 25,
            'conv5_1': 28, 'conv5_2': 30, 'conv5_3': 32, 'conv5_4': 34,
        }
        max_idx = conv_indices[layer]
        # Slice up to the convolution (pre-ReLU)
        self.features = nn.Sequential(*[vgg[i] for i in range(max_idx + 1)])
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.eval()

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def _preprocess(self, x_1ch: torch.Tensor) -> torch.Tensor:
        # Inputs are in [-1,1] with shape [B,1,H,W]; convert to 3ch and normalize
        x = (x_1ch + 1.0) * 0.5  # to [0,1]
        x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        return x

    def forward(self, x_1ch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self._preprocess(x_1ch)
        return self.features(x)


class PerceptualLoss(nn.Module):
    def __init__(self, layer: str = 'conv5_4', loss_fn: str = 'l1'):
        super().__init__()
        self.extractor = VGG19FeatureExtractor(layer)
        self.criterion = nn.L1Loss() if loss_fn == 'l1' else nn.MSELoss()

    def forward(self, sr_1ch: torch.Tensor, hr_1ch: torch.Tensor) -> torch.Tensor:
        self.extractor.eval()
        with torch.no_grad():
            feat_hr = self.extractor(hr_1ch)
        feat_sr = self.extractor(sr_1ch)  # gradients only through SR path
        return self.criterion(feat_sr, feat_hr)


# -----------------------------
# RaGAN Loss
# -----------------------------
class RaGANLoss:
    """
    Implements relativistic average GAN losses for discriminator and generator.
    Works with patch-based logits maps.
    """
    # Logit = raw, unnormalized output before the Sigmoid function
    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        real_mean = fake_logits.detach().mean()
        fake_mean = real_logits.detach().mean()
        loss_real = F.binary_cross_entropy_with_logits(real_logits - real_mean, torch.ones_like(real_logits))
        loss_fake = F.binary_cross_entropy_with_logits(fake_logits - fake_mean, torch.zeros_like(fake_logits))
        return loss_real + loss_fake

    def g_loss(self, real_logits_detached: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        real_mean = fake_logits.mean()
        fake_mean = real_logits_detached.mean()
        loss_real = F.binary_cross_entropy_with_logits(real_logits_detached - real_mean, torch.zeros_like(real_logits_detached))
        loss_fake = F.binary_cross_entropy_with_logits(fake_logits - fake_mean, torch.ones_like(fake_logits))
        return loss_real + loss_fake


# -----------------------------
# Data
# -----------------------------
def _load_split_from_json(split_json: str) -> Dict[str, list]:
    with open(split_json, 'r') as f:
        payload = json.load(f)
    result = {k: [] for k in ['train', 'val', 'test']}
    for split_name in result.keys():
        if split_name in payload.get('splits', {}):
            result[split_name] = [entry.get('path') for entry in payload['splits'][split_name] if 'path' in entry]
    return result


def build_dataloaders(root: str, scale: int, batch_size: int, patch_size: int, num_workers: int = 4, split_json: str = None,
                      degradation: str = 'blurnoise', blur_sigma_range=None, blur_kernel: int = None,
                      noise_sigma_range_norm=(0.001, 0.003), dose_factor_range=(0.25, 0.5), antialias_clean: bool = True) -> Tuple[DataLoader, DataLoader]:
    if split_json and os.path.isfile(split_json):
        print(f"[Split] Using split mapping from {split_json}")
        splits = _load_split_from_json(split_json)
        # Confirm loaded counts
        print(f"[Split] Loaded counts: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    else:
        print("[Split] No valid split_json provided; generating 70/15/15 split with seed=42")
        splits = split_patients(root, seed=42)

    train_dirs = splits['train']
    val_dirs = splits['val']

    # Filter non-existing paths defensively (especially when loading from JSON)
    train_dirs = [p for p in train_dirs if isinstance(p, str) and os.path.isdir(p)]
    val_dirs = [p for p in val_dirs if isinstance(p, str) and os.path.isdir(p)]
    print(f"[Split] Existing dirs after filtering: train={len(train_dirs)} val={len(val_dirs)}")

    train_ds = ConcatDataset([
        CT_Dataset_SR(
            d,
            scale_factor=scale,
            do_random_crop=True,
            hr_patch=patch_size,
            degradation=degradation,
            blur_sigma_range=blur_sigma_range,
            blur_kernel=blur_kernel,
            noise_sigma_range_norm=tuple(noise_sigma_range_norm),
            dose_factor_range=tuple(dose_factor_range),
            antialias_clean=antialias_clean
        ) for d in train_dirs
    ])

    


    val_ds = ConcatDataset([
        CT_Dataset_SR(
            d,
            scale_factor=scale,
            do_random_crop=False,
            degradation=degradation,
            blur_sigma_range=blur_sigma_range,
            blur_kernel=blur_kernel,
            noise_sigma_range_norm=tuple(noise_sigma_range_norm),
            dose_factor_range=tuple(dose_factor_range),
            antialias_clean=antialias_clean,
            degradation_sampling='volume',
            deg_seed=fixed_seed_for_path(d, base=42) # Deterministic seeds per validation patient for reproducibility
        ) for d in val_dirs
    ])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, persistent_workers=False)
    # Validation on whole slices (variable sizes) → batch_size=1 to avoid collate size mismatches
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=max(1, num_workers // 2),
                            pin_memory=True, persistent_workers=(num_workers // 2) > 0)
    return train_loader, val_loader


# -----------------------------
# Models
# -----------------------------
def build_models(scale: int, pretrained_g: str = None, device: torch.device = torch.device('cpu')) -> Tuple[nn.Module, nn.Module, EMA]:
    G = RRDBNet_CT(in_nc=1, out_nc=1, scale=scale).to(device)
    if pretrained_g and os.path.isfile(pretrained_g):
        sd = torch.load(pretrained_g, map_location=device)
        if isinstance(sd, dict) and 'model' in sd and all(k in sd for k in ['epoch', 'model']):
            print(f"[Init] Detected checkpoint dict; loading weights from 'model' key: {pretrained_g}")
            sd = sd['model']
        G.load_state_dict(sd, strict=True)
        print(f"[Init] Loaded pretrained G weights from {pretrained_g}")
    else:
        print("[Init] Training G from provided weights (or randomly if path invalid)")

    D = PatchDiscriminatorSN(in_nc=1).to(device)
    ema = EMA(G, decay=0.999)
    ema.ema_model.to(device)
    return G, D, ema


# -----------------------------
# Train / Validate
# -----------------------------
def validate(G_ema: nn.Module, val_loader: DataLoader, device: torch.device, *,
             perceptual_loss: 'PerceptualLoss', ragan: 'RaGANLoss', D: nn.Module = None,
             lambda_pix: float = 1.0, lambda_perc: float = 0.10, lambda_gan: float = 0.005,
             compute_gan: bool = False) -> Dict[str, float]:
    G_ema.eval()
    if D is not None:
        D.eval()
    n = 0
    # accumulators
    val_l1 = 0.0
    val_perc = 0.0
    val_gan = 0.0
    metrics_accum = {k: 0.0 for k in ['MAE', 'MSE', 'RMSE', 'PSNR', 'SSIM']}
    use_cuda = (device.type == 'cuda')
    l1 = nn.L1Loss()
    with torch.no_grad():
        for lr, hr in val_loader:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            with torch_amp.autocast('cuda', enabled=use_cuda):
                sr = G_ema(lr)
                lpix = l1(sr, hr).item()
                lperc = perceptual_loss(sr, hr).item()
                if compute_gan and (D is not None):
                    real_logits = D(hr)
                    fake_logits = D(sr)
                    lgan = ragan.g_loss(real_logits, fake_logits).item()
                else:
                    lgan = 0.0
            val_l1 += float(lpix)
            val_perc += float(lperc)
            val_gan += float(lgan)

            device_str = 'cuda' if use_cuda else 'cpu'
            if sr.ndim == 4:
                batch = sr.shape[0]
                for i in range(batch):
                    m = evaluate_metrics(sr[i].detach(), hr[i].detach(), metrics_device=device_str)
                    for k in metrics_accum:
                        metrics_accum[k] += float(m[k])
                n += batch
            else:
                m = evaluate_metrics(sr.detach(), hr.detach(), metrics_device=device_str)
                for k in metrics_accum:
                    metrics_accum[k] += float(m[k])
                n += 1

    inv_n = 1.0 / max(1, n)
    for k in metrics_accum:
        metrics_accum[k] *= inv_n
    val_l1 *= inv_n
    val_perc *= inv_n
    val_gan *= inv_n
    val_total = lambda_pix * val_l1 + lambda_perc * val_perc + lambda_gan * val_gan
    val_total_no_gan = lambda_pix * val_l1 + lambda_perc * val_perc
    return {
        'L1': val_l1,
        'Perceptual': val_perc,
        'GAN': val_gan,
        'Total': val_total,
        'Total_NoGAN': val_total_no_gan,
        'W_L1': lambda_pix * val_l1,
        'W_Perc': lambda_perc * val_perc,
        'W_GAN': lambda_gan * val_gan,
        'MAE': metrics_accum['MAE'],
        'MSE': metrics_accum['MSE'],
        'RMSE': metrics_accum['RMSE'],
        'PSNR': metrics_accum['PSNR'],
        'SSIM': metrics_accum['SSIM'],
    }


def train_one_epoch(
    G: nn.Module,
    D: nn.Module,
    ema: EMA,
    train_loader: DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    scaler: torch_amp.GradScaler,
    perceptual_loss: PerceptualLoss,
    ragan: RaGANLoss,
    device: torch.device,
    lambda_pix: float = 1.0,
    lambda_perc: float = 0.10,
    lambda_gan: float = 0.005,
    log_interval: int = 100,
    warmup_g_only_iters: int = 0,
) -> Dict[str, float]:
    G.train()
    D.train()
    l1 = nn.L1Loss()
    running_total = 0.0
    running_l1 = 0.0
    running_perc = 0.0
    running_gan = 0.0
    it = 0

    for lr, hr in train_loader:
        it += 1
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        # -----------------
        # Update Discriminator (skip in warmup)
        # -----------------
        if it > warmup_g_only_iters:
            optimizer_d.zero_grad(set_to_none=True)
            use_cuda = (device.type == 'cuda')
            with torch_amp.autocast('cuda', enabled=use_cuda):
                with torch.no_grad():
                    sr = G(lr)
                real_logits = D(hr)
                fake_logits = D(sr.detach())
                d_loss = ragan.d_loss(real_logits, fake_logits)
            scaler.scale(d_loss).backward()
            nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            scaler.step(optimizer_d)
        else:
            d_loss = torch.tensor(0.0, device=device)

        # -----------------
        # Update Generator
        # -----------------
        optimizer_g.zero_grad(set_to_none=True)
        use_cuda = (device.type == 'cuda')
        with torch_amp.autocast('cuda', enabled=use_cuda):
            sr = G(lr)
            l_pix = l1(sr, hr)
            l_perc = perceptual_loss(sr, hr)
            fake_logits = D(sr)
            with torch.no_grad():
                real_logits_detached = D(hr).detach()
            l_gan = ragan.g_loss(real_logits_detached, fake_logits)
            total = lambda_pix * l_pix + lambda_perc * l_perc + lambda_gan * l_gan
        scaler.scale(total).backward()
        nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
        scaler.step(optimizer_g)
        scaler.update()

        # EMA update after G step
        ema.update(G)

        running_total += float(total.detach().cpu())
        running_l1 += float(l_pix.detach().cpu())
        running_perc += float(l_perc.detach().cpu())
        running_gan += float(l_gan.detach().cpu())

        if it % log_interval == 0:
            d_real = torch.sigmoid(real_logits.detach()).mean().item() if it > warmup_g_only_iters else 0.0
            d_fake = torch.sigmoid(fake_logits.detach()).mean().item()
            current_lr = optimizer_g.param_groups[0]['lr']
            print(f"  [Iter {it}] L_pix={l_pix.item():.4f} L_perc={l_perc.item():.4f} L_gan={l_gan.item():.4f} L_total={total.item():.4f} | D_real={d_real:.3f} D_fake={d_fake:.3f} | lr={current_lr:.6f}")

        # memory cleanup hints
        del sr, l_pix, l_perc, l_gan, total, fake_logits
        if 'real_logits' in locals():
            del real_logits

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    inv = 1.0 / max(1, it)
    avg_l1 = running_l1 * inv
    avg_perc = running_perc * inv
    avg_gan = running_gan * inv
    total_no_gan = (1.0 * avg_l1) + (lambda_perc * avg_perc)
    return {
        'total': running_total * inv,
        'total_no_gan': total_no_gan,
        'l1': avg_l1,
        'perc': avg_perc,
        'gan': avg_gan,
        'w_l1': 1.0 * avg_l1,
        'w_perc': lambda_perc * avg_perc,
        'w_gan': lambda_gan * avg_gan,
    }


# -----------------------------
# Checkpointing / Plotting
# -----------------------------
def save_checkpoint(out_dir: str,
                    G_ema: nn.Module,
                    G_live: nn.Module,
                    D: nn.Module,
                    optimizer_g: torch.optim.Optimizer,
                    optimizer_d: torch.optim.Optimizer,
                    scaler: torch_amp.GradScaler,
                    scheduler_g: torch.optim.lr_scheduler._LRScheduler,
                    scheduler_d: torch.optim.lr_scheduler._LRScheduler,
                    epoch: int,
                    global_step: int,
                    tag: str,
                    *, metadata: dict = None,
                    ema_decay: float = 0.999,
                    best_psnr: float = None,
                    best_mae: float = None,
                    best_total_no_gan: float = None):
    """Save checkpoint with complete training state (EMA/live, G/D, optimizers, schedulers, scaler)."""
    os.makedirs(out_dir, exist_ok=True)

    path_ema = os.path.join(out_dir, f'{tag}.pth')
    payload_common = {
        'epoch': int(epoch),
        'global_step': int(global_step),
        'D': D.state_dict(),
        'optimizer_g': optimizer_g.state_dict(),
        'optimizer_d': optimizer_d.state_dict(),
        'scaler': (scaler.state_dict() if (scaler is not None and isinstance(scaler, torch_amp.GradScaler)) else None),
        'scheduler_g': scheduler_g.state_dict() if scheduler_g is not None else None,
        'scheduler_d': scheduler_d.state_dict() if scheduler_d is not None else None,
        'ema_decay': float(ema_decay),
        'meta': metadata,
        'best_psnr': float(best_psnr) if best_psnr is not None else None,
        'best_mae': float(best_mae) if best_mae is not None else None,
        'best_total_no_gan': float(best_total_no_gan) if best_total_no_gan is not None else None,
    }
    payload_ema = {
        **payload_common,
        'model': G_ema.state_dict(),
        'ema_model': G_ema.state_dict(),
        'weights_type': 'ema'
    }
    torch.save(payload_ema, path_ema)
    print(f"[CKPT] Saved {tag} (EMA) -> {path_ema}")

    path_live = os.path.join(out_dir, f'{tag}_live.pth')
    payload_live = {
        **payload_common,
        'model': G_live.state_dict(),
        'ema_model': G_ema.state_dict(),
        'weights_type': 'live'
    }
    torch.save(payload_live, path_live)
    print(f"[CKPT] Saved {tag} (Live) -> {path_live}")


def plot_curves(history: Dict[str, list], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # Total loss curves (with GAN if computed in val; otherwise val_total equals no-GAN)
    try:
        plt.figure(figsize=(8,5))
        if 'train_total' in history:
            plt.plot(history['train_total'], label='Train Total')
        if 'val_total' in history:
            plt.plot(history['val_total'], label='Val Total')
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'loss_total.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"[Plot] Could not save loss_total.png: {e}")

    # Total loss curves without GAN
    try:
        if 'train_total_no_gan' in history or 'val_total_no_gan' in history:
            plt.figure(figsize=(8,5))
            if 'train_total_no_gan' in history:
                plt.plot(history['train_total_no_gan'], label='Train Total (no GAN)')
            if 'val_total_no_gan' in history:
                plt.plot(history['val_total_no_gan'], label='Val Total (no GAN)')
            plt.xlabel('Epoch')
            plt.ylabel('Total Loss (no GAN)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'loss_total_nogan.png'), dpi=150)
            plt.close()
    except Exception as e:
        print(f"[Plot] Could not save loss_total_nogan.png: {e}")

    # Unweighted components curves (kept for backward-compat)
    try:
        plt.figure(figsize=(10,6))
        for key, label in [
            ('train_l1','Train L1'), ('train_perc','Train Perceptual'), ('train_gan','Train GAN'),
            ('val_l1','Val L1'), ('val_perc','Val Perceptual'), ('val_gan','Val GAN')
        ]:
            if key in history and len(history[key]) > 0:
                plt.plot(history[key], label=label)
        plt.xlabel('Epoch')
        plt.ylabel('Loss Components')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'loss_components.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"[Plot] Could not save loss_components.png: {e}")

    # Weighted components curves
    try:
        plt.figure(figsize=(10,6))
        for key, label in [
            ('train_w_l1','Train wL1'), ('train_w_perc','Train wPerc'), ('train_w_gan','Train wGAN'),
            ('val_w_l1','Val wL1'), ('val_w_perc','Val wPerc'), ('val_w_gan','Val wGAN')
        ]:
            if key in history and len(history[key]) > 0:
                plt.plot(history[key], label=label)
        plt.xlabel('Epoch')
        plt.ylabel('Weighted Components')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'loss_components_weighted.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"[Plot] Could not save loss_components_weighted.png: {e}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description='Finetune RRDB on CT with ESRGAN objectives (conservative)')
    parser.add_argument('--data_root', type=str, default='preprocessed_data', help='Root with patient subfolders (default: ESRGAN/preprocessed_data)')
    parser.add_argument('--scale', type=int, default=2, choices=[2,4])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--patch', type=int, default=192)
    parser.add_argument('--pretrained_g', type=str, required=True)
    # removed out_dir (not used for new artifacts)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint (ema/live) to resume training')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_perc', type=float, default=0.08)
    parser.add_argument('--lambda_gan', type=float, default=0.003)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--warmup_g_only', type=int, default=100, help='number of iterations to train G only at start')
    parser.add_argument('--split_json', type=str, default=None, help='Path to patient split JSON (from dump_patient_split.py)')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience in epochs (None disables)')
    parser.add_argument('--early_metric', type=str, default='total_no_gan', choices=['mae','psnr','total_no_gan'], help='Metric to monitor for early stopping')
    # Degradation options
    parser.add_argument('--degradation', type=str, default='blurnoise', choices=['clean', 'blur', 'blurnoise'], help='Degradation pipeline for LR generation')
    parser.add_argument('--blur_sigma_range', type=float, nargs=2, default=None, help='Range [lo hi] of Gaussian blur sigma; if None, defaults by scale')
    parser.add_argument('--blur_kernel', type=int, default=None, help='Explicit odd kernel size; if None, derived from sigma')
    parser.add_argument('--noise_sigma_range_norm', type=float, nargs=2, default=[0.001, 0.003], help='Gaussian noise sigma range on normalized [-1,1] image')
    parser.add_argument('--dose_factor_range', type=float, nargs=2, default=[0.25, 0.5], help='Dose factor range; noise scales ~ 1/sqrt(dose)')
    parser.add_argument('--antialias_clean', action='store_true', help='Use antialias in clean downsample')
    parser.add_argument('--val_compute_gan', action='store_true', help='Wenn gesetzt, wird der GAN-Term in der Validierung mitberechnet und in Total/Komponenten eingerechnet.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using {device}")

    # Data
    train_loader, val_loader = build_dataloaders(
        root=args.data_root, scale=args.scale, batch_size=args.batch_size, patch_size=args.patch, num_workers=args.num_workers,
        split_json=args.split_json,
        degradation=args.degradation,
        blur_sigma_range=args.blur_sigma_range,
        blur_kernel=args.blur_kernel,
        noise_sigma_range_norm=args.noise_sigma_range_norm,
        dose_factor_range=args.dose_factor_range,
        antialias_clean=args.antialias_clean
    )

    # Runs folder & tee logger
    exp_name = f"finetune_x{args.scale}_{args.degradation}"
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join('runs', f"{exp_name}_{timestamp}")
    _ensure_dir(run_dir)
    log_path = os.path.join(run_dir, 'train.log')
    sys.stdout = _Tee(log_path)

    # Models
    G, D, ema = build_models(scale=args.scale, pretrained_g=args.pretrained_g, device=device)
    perceptual = PerceptualLoss(layer='conv5_4').to(device)
    ragan = RaGANLoss()

    optimizer_g = Adam(G.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    optimizer_d = Adam(D.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)

    milestones = [max(1, int(args.epochs * 0.6)), max(2, int(args.epochs * 0.8))]
    scheduler_g = MultiStepLR(optimizer_g, milestones=milestones, gamma=0.5)
    scheduler_d = MultiStepLR(optimizer_d, milestones=milestones, gamma=0.5)

    # Echo CLI config
    print("[Config] data_root=", args.data_root)
    print("[Config] scale=", args.scale)
    print("[Config] epochs=", args.epochs)
    print("[Config] batch_size=", args.batch_size)
    print("[Config] patch=", args.patch)
    print("[Config] pretrained_g=", args.pretrained_g)
    print("[Config] lr=", args.lr)
    print("[Config] lambda_perc=", args.lambda_perc)
    print("[Config] lambda_gan=", args.lambda_gan)
    print("[Config] num_workers=", args.num_workers)
    print("[Config] warmup_g_only=", args.warmup_g_only)
    print("[Config] split_json=", args.split_json)
    print("[Config] patience=", args.patience)
    print("[Config] early_metric=", args.early_metric)
    print("[Config] degradation=", args.degradation)
    print("[Config] blur_sigma_range=", args.blur_sigma_range)
    print("[Config] blur_kernel=", args.blur_kernel)
    print("[Config] noise_sigma_range_norm=", args.noise_sigma_range_norm)
    print("[Config] dose_factor_range=", args.dose_factor_range)
    print("[Config] antialias_clean=", args.antialias_clean)

    use_cuda = (device.type == 'cuda')
    scaler = torch_amp.GradScaler(enabled=use_cuda)

    history = {
        'train_total': [], 'train_total_no_gan': [], 'train_l1': [], 'train_perc': [], 'train_gan': [],
        'train_w_l1': [], 'train_w_perc': [], 'train_w_gan': [],
        'val_total': [], 'val_total_no_gan': [], 'val_l1': [], 'val_perc': [], 'val_gan': [],
        'val_w_l1': [], 'val_w_perc': [], 'val_w_gan': [],
        'val_psnr': [], 'val_mae': []
    }
    exp_name = f"rrdb_x{args.scale}_{args.degradation}"
    meta = {
        'experiment': exp_name,
        'scale_factor': args.scale,
        'degradation': args.degradation,
        'blur_sigma_range': args.blur_sigma_range if args.blur_sigma_range is not None else (None),
        'blur_kernel': args.blur_kernel,
        'noise_sigma_range_norm': args.noise_sigma_range_norm,
        'dose_factor_range': args.dose_factor_range,
        'notes': 'blur/noise degrader, jitter per patch (finetune)'
    }
    # Effective defaults for blur sigma/kernel
    if args.degradation in ('blur','blurnoise'):
        if args.blur_sigma_range is None:
            base_sigma = 0.8 if int(args.scale) == 2 else (1.2 if int(args.scale) == 4 else 0.8)
            jitter = 0.1 if int(args.scale) == 2 else 0.15
            eff_lo = max(1e-6, base_sigma - jitter)
            eff_hi = base_sigma + jitter
        else:
            eff_lo = float(args.blur_sigma_range[0])
            eff_hi = float(args.blur_sigma_range[1])
        blur_sigma_range_eff = [eff_lo, eff_hi]
        if args.blur_kernel is not None:
            blur_kernel_eff = int(args.blur_kernel)
        else:
            mid_sigma = 0.5 * (eff_lo + eff_hi)
            k = int(max(3, round(6.0 * float(mid_sigma))))
            blur_kernel_eff = k if (k % 2 == 1) else (k + 1)
    else:
        blur_sigma_range_eff = None
        blur_kernel_eff = None

    # write metadata JSON to run_dir
    try:
        cfg = {
            'experiment': exp_name,
            'run_dir': run_dir,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'amp_enabled': bool(use_cuda),
            'optimizer_G': {'name': 'Adam', 'lr': args.lr, 'betas': [0.9, 0.999]},
            'optimizer_D': {'name': 'Adam', 'lr': args.lr, 'betas': [0.9, 0.999]},
            'scheduler': {'type': 'MultiStepLR', 'milestones': milestones, 'gamma': 0.5},
            'scale_factor': args.scale,
            'patch': args.patch,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'warmup_g_only': args.warmup_g_only,
            'lambda_perc': args.lambda_perc,
            'lambda_gan': args.lambda_gan,
            'degradation': args.degradation,
            'blur_sigma_range': blur_sigma_range_eff,
            'blur_kernel': blur_kernel_eff,
            'noise_sigma_range_norm': args.noise_sigma_range_norm,
            'dose_factor_range': args.dose_factor_range,
            'degradation_sampling': {
                'train': 'volume (per-epoch resample)',
                'val': 'volume (fixed per patient)'
            },
            'splits': {
                'n_train_vols': len(train_loader.dataset.datasets) if isinstance(train_loader.dataset, ConcatDataset) else -1,
                'n_val_vols': len(val_loader.dataset.datasets) if isinstance(val_loader.dataset, ConcatDataset) else -1
            },
            'notes': 'ESRGAN finetune with EMA',
        }
        if args.resume:
            cfg['resume_path'] = args.resume
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)
        print(f"[Meta] Wrote config -> {os.path.join(run_dir, 'config.json')}")
    except Exception as e:
        print(f"[Meta] Could not write config: {e}")
    best_psnr = -1e9
    best_mae = 1e9
    best_total_no_gan = float('inf')
    epochs_no_improve = 0

    # Central monitor configuration for early metric
    metric_cfg = {
        'mae':         {'key': 'MAE',         'mode': 'min', 'eps': 1e-8},
        'psnr':        {'key': 'PSNR',        'mode': 'max', 'eps': 1e-6},
        'total_no_gan':{'key': 'Total_NoGAN', 'mode': 'min', 'eps': 1e-9},
    }
    mon = metric_cfg[args.early_metric]
    monitor_key, mode, eps = mon['key'], mon['mode'], mon['eps']

    def is_improved(curr: float, best: float, mode: str, eps: float) -> bool:
        if mode == 'min':
            return curr < (best - eps)
        else:
            return curr > (best + eps)

    def get_best_for_early_metric():
        if args.early_metric == 'mae':
            return best_mae
        elif args.early_metric == 'psnr':
            return best_psnr
        else:
            return best_total_no_gan

    # CSV metrics in run_dir
    csv_path = os.path.join(run_dir, 'metrics.csv')
    if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch','global_step',
                             'train_total','train_total_no_gan','train_l1','train_perc','train_gan','train_w_l1','train_w_perc','train_w_gan',
                             'val_total','val_total_no_gan','val_l1','val_perc','val_gan','val_w_l1','val_w_perc','val_w_gan',
                             'psnr','mae','lr','time_sec'])

    # Resume training if provided
    start_epoch = 1
    iters_seen = 0
    if args.resume and os.path.isfile(args.resume):
        try:
            ckpt = torch.load(args.resume, map_location=device)
            if isinstance(ckpt, dict):
                # Prefer live weights when resuming
                if 'model' in ckpt:
                    G.load_state_dict(ckpt['model'], strict=True)
                if 'ema_model' in ckpt:
                    try:
                        ema.ema_model.load_state_dict(ckpt['ema_model'], strict=True)
                    except Exception:
                        pass
                if 'D' in ckpt:
                    D.load_state_dict(ckpt['D'], strict=True)
                if 'optimizer_g' in ckpt:
                    optimizer_g.load_state_dict(ckpt['optimizer_g'])
                if 'optimizer_d' in ckpt:
                    optimizer_d.load_state_dict(ckpt['optimizer_d'])
                if 'scheduler_g' in ckpt and ckpt['scheduler_g'] is not None:
                    try:
                        scheduler_g.load_state_dict(ckpt['scheduler_g'])
                    except Exception:
                        pass
                if 'scheduler_d' in ckpt and ckpt['scheduler_d'] is not None:
                    try:
                        scheduler_d.load_state_dict(ckpt['scheduler_d'])
                    except Exception:
                        pass
                if use_cuda and ('scaler' in ckpt and ckpt['scaler'] is not None):
                    try:
                        scaler.load_state_dict(ckpt['scaler'])
                    except Exception:
                        pass
                if 'epoch' in ckpt:
                    start_epoch = int(ckpt['epoch']) + 1
                iters_seen = int(ckpt.get('global_step', 0))
                best_psnr = float(ckpt.get('best_psnr', best_psnr))
                best_mae = float(ckpt.get('best_mae', best_mae))
                if 'best_total_no_gan' in ckpt and ckpt['best_total_no_gan'] is not None:
                    try:
                        best_total_no_gan = float(ckpt['best_total_no_gan'])
                    except Exception:
                        pass
                print(f"[Resume] Resumed from {args.resume} at epoch={start_epoch} global_step={iters_seen}")
        except Exception as e:
            print(f"[Resume] Failed to load resume checkpoint: {e}")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"[Epoch {epoch}/{args.epochs}] Starting ...")
        # Per-epoch resample for training datasets (volume-wise)
        subdatasets = getattr(train_loader.dataset, 'datasets', None)
        if subdatasets is not None:
            for ds in subdatasets:
                if hasattr(ds, 'resample_volume_params'):
                    ds.resample_volume_params(epoch_seed=epoch)
            print(f"[Deg-Resample] Training volumes resampled for epoch {epoch}")
        else:
            if hasattr(train_loader.dataset, 'resample_volume_params'):
                train_loader.dataset.resample_volume_params(epoch_seed=epoch)
                print(f"[Deg-Resample] Training volumes resampled for epoch {epoch}")
        t0 = time.perf_counter()
        warmup_iters_remaining = max(0, args.warmup_g_only - iters_seen)
        train_out = train_one_epoch(
            G, D, ema, train_loader, optimizer_g, optimizer_d, scaler, perceptual, ragan, device,
            lambda_pix=1.0, lambda_perc=args.lambda_perc, lambda_gan=args.lambda_gan,
            log_interval=100, warmup_g_only_iters=warmup_iters_remaining
        )
        history['train_total'].append(train_out['total'])
        history['train_total_no_gan'].append(train_out['total_no_gan'])
        history['train_l1'].append(train_out['l1'])
        history['train_perc'].append(train_out['perc'])
        history['train_gan'].append(train_out['gan'])
        history['train_w_l1'].append(train_out['w_l1'])
        history['train_w_perc'].append(train_out['w_perc'])
        history['train_w_gan'].append(train_out['w_gan'])

        # We can estimate how many iters happened
        iters_seen += len(train_loader)

        # Scheduler steps per epoch
        scheduler_g.step()
        scheduler_d.step()

        # Validation using EMA
        val_metrics = validate(ema.ema_model, val_loader, device,
                               perceptual_loss=perceptual, ragan=ragan, D=D,
                               lambda_pix=1.0, lambda_perc=args.lambda_perc, lambda_gan=args.lambda_gan,
                               compute_gan=args.val_compute_gan)
        history['val_total'].append(val_metrics['Total'])
        history['val_total_no_gan'].append(val_metrics['Total_NoGAN'])
        history['val_l1'].append(val_metrics['L1'])
        history['val_perc'].append(val_metrics['Perceptual'])
        history['val_gan'].append(val_metrics['GAN'])
        history['val_w_l1'].append(val_metrics['W_L1'])
        history['val_w_perc'].append(val_metrics['W_Perc'])
        history['val_w_gan'].append(val_metrics['W_GAN'])
        if not args.val_compute_gan and epoch == start_epoch:
            print("[Val] GAN term disabled (use --val_compute_gan to enable).")
        print(f"[Val] Total={val_metrics['Total']:.6f} | Total_NoGAN={val_metrics['Total_NoGAN']:.6f} | L1={val_metrics['L1']:.6f} Perc={val_metrics['Perceptual']:.6f} GAN={val_metrics['GAN']:.6f} | PSNR={val_metrics['PSNR']:.4f} SSIM={val_metrics['SSIM']:.4f}")

        elapsed = time.perf_counter() - t0
        current_lr = optimizer_g.param_groups[0]['lr']
        try:
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, iters_seen,
                    f"{train_out['total']:.6f}", f"{train_out['total_no_gan']:.6f}", f"{train_out['l1']:.6f}", f"{train_out['perc']:.6f}", f"{train_out['gan']:.6f}", f"{train_out['w_l1']:.6f}", f"{train_out['w_perc']:.6f}", f"{train_out['w_gan']:.6f}",
                    f"{val_metrics['Total']:.6f}", f"{val_metrics['Total_NoGAN']:.6f}", f"{val_metrics['L1']:.6f}", f"{val_metrics['Perceptual']:.6f}", f"{val_metrics['GAN']:.6f}", f"{val_metrics['W_L1']:.6f}", f"{val_metrics['W_Perc']:.6f}", f"{val_metrics['W_GAN']:.6f}",
                    f"{val_metrics['PSNR']:.4f}", f"{val_metrics['MAE']:.6f}", f"{current_lr:.6f}", f"{elapsed:.2f}"
                ])
        except Exception as e:
            print(f"[CSV] Could not append metrics: {e}")

        # Monitoring & Checkpoints controlled only by selected early metric
        curr = float(val_metrics[monitor_key])
        best_curr = float(get_best_for_early_metric())
        improved_early = is_improved(curr, best_curr, mode, eps)
        print(f"[Monitor] early_metric={args.early_metric} curr={curr:.6f} best={best_curr:.6f} delta={(curr-best_curr):+.6f} mode={mode}")

        if improved_early:
            if args.early_metric == 'mae':
                best_mae = curr
            elif args.early_metric == 'psnr':
                best_psnr = curr
            else:
                best_total_no_gan = curr
            save_checkpoint(
                run_dir, ema.ema_model, G, D,
                optimizer_g, optimizer_d, scaler,
                scheduler_g, scheduler_d,
                epoch, iters_seen,
                tag='best', metadata=meta, ema_decay=ema.decay,
                best_psnr=best_psnr, best_mae=best_mae, best_total_no_gan=best_total_no_gan
            )
        save_checkpoint(
            run_dir, ema.ema_model, G, D,
            optimizer_g, optimizer_d, scaler,
            scheduler_g, scheduler_d,
            epoch, iters_seen,
            tag='last', metadata=meta, ema_decay=ema.decay,
            best_psnr=best_psnr, best_mae=best_mae, best_total_no_gan=best_total_no_gan
        )

        # Early stopping uses the same improvement decision
        if args.patience is not None:
            if improved_early:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"[EarlyStop] No improvement for {epochs_no_improve}/{args.patience} epochs on {args.early_metric}")
                if epochs_no_improve >= args.patience:
                    print("[EarlyStop] Patience reached. Stopping training early.")
                    break

    # Plots
    plot_curves(history, run_dir)


if __name__ == '__main__':
    main()


