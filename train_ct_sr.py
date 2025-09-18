import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from rrdb_ct_model import RRDBNet_CT
from ct_dataset_loader import CT_Dataset_SR
from seed_utils import fixed_seed_for_path
import os
import sys
import time
import csv
from datetime import datetime
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
from torch import amp
import hashlib
import random

#AMP (Automatic Mixed Precision) is used --> Operations run internally in float16 and not 32, which saves memory
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    use_cuda = (device.type == 'cuda')
    with torch.no_grad():
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs = lr_imgs.to(device, non_blocking=True) # Pytorch can start transfer asynchronously-->while the GPU is still computing, the next batch can be copied 
            hr_imgs = hr_imgs.to(device, non_blocking=True)
            with amp.autocast('cuda', enabled=use_cuda): #Tensor Cores compute in float16
                preds = model(lr_imgs)
                loss = criterion(preds, hr_imgs)
            total_loss += loss.item()
    model.train()
    return total_loss / max(1, len(dataloader))

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)


class _Tee:
    def __init__(self, log_path):
        self._stdout = sys.stdout
        # line-buffered
        self._fh = open(log_path, 'a', encoding='utf-8', buffering=1)
    def write(self, data):
        self._stdout.write(data)
        self._fh.write(data)
    def flush(self):
        self._stdout.flush()
        self._fh.flush()


# Training function
def train_sr_model(model, train_loader, val_loader, num_epochs=20, lr=1e-4, patience=5,
                   run_dir=None, resume_path=None, *, metadata: dict = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr) # Betas: 0.9, 0.999 (default)
    criterion = nn.L1Loss()  
    use_cuda = (device.type == 'cuda')

    # Paths inside run_dir
    _ensure_dir(run_dir)
    ckpt_best = os.path.join(run_dir, 'best.pth')
    ckpt_last = os.path.join(run_dir, 'last.pth')
    csv_path  = os.path.join(run_dir, 'metrics.csv')
    plot_path = os.path.join(run_dir, 'training_curve.png')

    # CSV header
    if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'global_step', 'train_l1', 'val_l1', 'lr', 'time_sec'])

    # Resume support
    best_val = float('inf')
    global_step = 0
    start_epoch = 1
    last_epoch = start_epoch - 1
    scaler = amp.GradScaler(enabled=use_cuda)
    if resume_path is not None and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            try:
                model.load_state_dict(ckpt['model'])
                if 'optimizer_g' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer_g'])
                if use_cuda and ('scaler' in ckpt and ckpt['scaler'] is not None):
                    try:
                        scaler.load_state_dict(ckpt['scaler'])
                    except Exception:
                        pass
                best_val = float(ckpt.get('best_val', best_val))
                # resume from next epoch after the one stored in checkpoint
                start_epoch = int(ckpt.get('epoch', 0)) + 1
                global_step = int(ckpt.get('global_step', 0))
                print(f"[Resume] Loaded checkpoint from {resume_path} | epoch={start_epoch} global_step={global_step} best_val={best_val:.6f}")
            except Exception as e:
                print(f"[Resume] Failed to load checkpoint: {e}")

    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(start_epoch, num_epochs + 1):
        # Resample degradation params for all training sub-datasets (ConcatDataset)
        subdatasets = getattr(train_loader.dataset, "datasets", None)
        if subdatasets is not None:
            for ds in subdatasets:
                if hasattr(ds, "resample_volume_params"):
                    ds.resample_volume_params(epoch_seed=epoch)
            print(f"[Deg-Resample] Training volumes resampled for epoch {epoch}")
        else:
            if hasattr(train_loader.dataset, "resample_volume_params"):
                train_loader.dataset.resample_volume_params(epoch_seed=epoch)
                print(f"[Deg-Resample] Training volumes resampled for epoch {epoch}")
        model.train()
        t0 = time.perf_counter()
        total_train = 0.0
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(train_loader, start=1):
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True) # Gradienten not 0, but None -> less memory accesses
            with amp.autocast('cuda', enabled=use_cuda):
                preds = model(lr_imgs)
                loss = criterion(preds, hr_imgs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train += loss.item()
            global_step += 1
            if batch_idx % 50 == 0:
                print(f"  [Batch {batch_idx}/{len(train_loader)}] train L1={loss.item():.5f}")
            
            del preds, loss
        torch.cuda.empty_cache()

        avg_train = total_train / max(1, len(train_loader))
        avg_val = validate(model, val_loader, criterion, device)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        elapsed = time.perf_counter() - t0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Train] Epoch {epoch}/{num_epochs} done | train L1: {avg_train:.6f} | val L1: {avg_val:.6f} | lr={current_lr:.6f} | time={elapsed:.1f}s")

        # Append CSV row
        try:
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, global_step, f"{avg_train:.6f}", f"{avg_val:.6f}", f"{current_lr:.6f}", f"{elapsed:.2f}"])
        except Exception as e:
            print(f"[CSV] Could not append metrics: {e}")

        # Early Stopping + save best
        if avg_val < best_val - 1e-6:
            best_val = avg_val
            epochs_no_improve = 0
            payload = {
                'epoch': epoch,
                'global_step': global_step,
                'model': model.state_dict(),
                'optimizer_g': optimizer.state_dict(),
                'scaler': scaler.state_dict() if use_cuda else None,
                'weights_type': 'pretrain_l1',
                'best_val': float(best_val),
                'meta': metadata
            }
            torch.save(payload, ckpt_best)
            print(f"[CKPT] Saved best -> {ckpt_best}")
        else:
            epochs_no_improve += 1
            print(f"[Train] No val improvement ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print(f"[Train] Early stopping. Best val: {best_val:.6f}")
                break

        # track last completed epoch
        last_epoch = epoch

    # Save last model
    payload_last = {
        'epoch': int(last_epoch),
        'global_step': global_step,
        'model': model.state_dict(),
        'optimizer_g': optimizer.state_dict(),
        'scaler': scaler.state_dict() if use_cuda else None,
        'weights_type': 'pretrain_l1',
        'best_val': float(best_val),
        'meta': metadata
    }
    torch.save(payload_last, ckpt_last)
    print(f"[CKPT] Saved last -> {ckpt_last}")

    # Save plot
    try:
        plt.figure(figsize=(7,4))
        plt.plot(range(1, len(train_losses)+1), train_losses, label='Train loss (L1)')
        plt.plot(range(1, len(val_losses)+1), val_losses, label='Val loss (L1)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[Plot] Saved training curve -> {plot_path}")
    except Exception as e:
        print(f"[Train] Could not save plot: {e}")

    return model, train_losses, val_losses


# Startpoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RRDBNet_CT on CT super-resolution with L1 loss (pretraining)')
    default_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preprocessed_data')
    parser.add_argument('--data_root', type=str, default=default_root, help='Root with patient subfolders (default: ESRGAN/preprocessed_data)')
    parser.add_argument('--scale', type=int, default=2, help='Upscaling factor (must divide patch_size)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Training batch size')
    parser.add_argument('--patch_size', type=int, default=192, help='HR patch size (must be divisible by scale)')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (epochs)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers for training')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='Global seed for reproducible runs')
    # Degradation options
    parser.add_argument('--degradation', type=str, default='blurnoise', choices=['clean', 'blur', 'blurnoise'], help='Degradation pipeline for LR generation')
    parser.add_argument('--blur_sigma_range', type=float, nargs=2, default=None, help='Range [lo hi] of Gaussian blur sigma (in normalized image units). If None, defaults by scale (x2≈0.8±0.1, x4≈1.2±0.15).')
    parser.add_argument('--blur_kernel', type=int, default=None, help='Explicit odd kernel size for blur. If None, derived from sigma')
    parser.add_argument('--noise_sigma_range_norm', type=float, nargs=2, default=[0.001, 0.003], help='Gaussian noise sigma range on normalized [-1,1] image (approx 0–10 HU)')
    parser.add_argument('--dose_factor_range', type=float, nargs=2, default=[0.25, 0.5], help='Dose factor range; noise scales ~ 1/sqrt(dose)')
    parser.add_argument('--antialias_clean', action='store_true', help='Use antialias in clean downsample')
    args = parser.parse_args()

    # Global seeding for reproducibility (before splits/model/loader)
    def set_global_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    set_global_seed(int(args.seed))

    if args.patch_size % args.scale != 0:
        raise ValueError(f"patch_size ({args.patch_size}) must be divisible by scale ({args.scale})")

    # Build run directory and tee logger
    exp_name = f"rrdb_x{args.scale}_{args.degradation}"
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join('runs', f"{exp_name}_{timestamp}")
    _ensure_dir(run_dir)
    # Tee logger
    log_path = os.path.join(run_dir, 'train.log')
    sys.stdout = _Tee(log_path)

    # Brief reproducibility log
    print(f"[Repro] seed={args.seed} | AMP={torch.cuda.is_available()}")
    print("[Repro] Train DataLoader uses a seeded torch.Generator for shuffle")

    print("[Args] Training configuration:")
    print(f"  data_root   : {args.data_root}")
    print(f"  scale       : {args.scale}")
    print(f"  epochs      : {args.epochs}")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  patch_size  : {args.patch_size}")
    print(f"  patience    : {args.patience}")
    print(f"  lr          : {args.lr}")
    print(f"  num_workers : {args.num_workers}")
    print(f"  degradation : {args.degradation}")
    print(f"  blur_sigma_range : {args.blur_sigma_range}")
    print(f"  blur_kernel : {args.blur_kernel}")
    print(f"  noise_sigma_range_norm : {args.noise_sigma_range_norm}")
    print(f"  dose_factor_range : {args.dose_factor_range}")
    print(f"  antialias_clean : {args.antialias_clean}")
    print(f"  resume      : {args.resume}")

    root = args.data_root
    patient_dirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if len(patient_dirs) == 0:
        raise RuntimeError(f"No patient directories found under {root}")

    perm = np.random.default_rng(42).permutation(len(patient_dirs)) # deterministic permutation to ensure same split across runs
    patient_dirs = [patient_dirs[i] for i in perm]
    n = len(patient_dirs)
    # 70/15/15 patient-wise split using deterministic index cutoffs
    train_cut = int(0.70 * n)
    val_cut = int(0.85 * n)
    train_dirs = patient_dirs[: train_cut]
    val_dirs   = patient_dirs[train_cut: val_cut]
    test_dirs  = patient_dirs[val_cut:]

    print(f"[Split] Patients total={n} | train={len(train_dirs)} | val={len(val_dirs)} | test={len(test_dirs)}")

    # Build datasets
    print("[Data] Building datasets ...")
    # Train on random, aligned patches; Val/Test on whole slices
    train_ds = ConcatDataset([
        CT_Dataset_SR(
            d,
            scale_factor=args.scale,
            do_random_crop=True,
            hr_patch=args.patch_size,
            hu_clip=(-1000, 2000),
            degradation=args.degradation,
            blur_sigma_range=args.blur_sigma_range,
            blur_kernel=args.blur_kernel,
            noise_sigma_range_norm=tuple(args.noise_sigma_range_norm),
            dose_factor_range=tuple(args.dose_factor_range),
            antialias_clean=args.antialias_clean
        ) for d in train_dirs
    ])
    

    val_ds   = ConcatDataset([
        CT_Dataset_SR(
            d,
            scale_factor=args.scale,
            do_random_crop=False,
            hu_clip=(-1000, 2000),
            degradation=args.degradation,
            blur_sigma_range=args.blur_sigma_range,
            blur_kernel=args.blur_kernel,
            noise_sigma_range_norm=tuple(args.noise_sigma_range_norm),
            dose_factor_range=tuple(args.dose_factor_range),
            antialias_clean=args.antialias_clean,
            degradation_sampling='volume',
            deg_seed=fixed_seed_for_path(d, base=42) # Validation/Test datasets with deterministic per-patient seeds (fixed across epochs)
        ) for d in val_dirs
    ])
    test_ds  = ConcatDataset([
        CT_Dataset_SR(
            d,
            scale_factor=args.scale,
            do_random_crop=False,
            hu_clip=(-1000, 2000),
            degradation=args.degradation,
            blur_sigma_range=args.blur_sigma_range,
            blur_kernel=args.blur_kernel,
            noise_sigma_range_norm=tuple(args.noise_sigma_range_norm),
            dose_factor_range=tuple(args.dose_factor_range),
            antialias_clean=args.antialias_clean,
            degradation_sampling='volume',
            deg_seed=fixed_seed_for_path(d, base=1337)
        ) for d in test_dirs
    ])

    # DataLoader
    print("[Data] Creating dataloaders ...")
    # Deterministic shuffle generator
    g = torch.Generator()
    g.manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False  # important for per-epoch Resampling
    )
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=max(1, args.num_workers//2))
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, num_workers=max(1, args.num_workers//2))

    # Initialize model
    print(f"[Init] Creating model RRDBNet_CT(scale={args.scale}) ...")
    model = RRDBNet_CT(scale=args.scale)

    # Metadata/config JSON (extended)
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp_enabled = torch.cuda.is_available()
    # Effective blur sigma range and kernel (fill defaults when not provided)
    if args.degradation in ('blur', 'blurnoise'):
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

    # Degradation-dependent metadata fields
    if args.degradation == 'blurnoise':
        noise_sigma_range_meta = args.noise_sigma_range_norm
        dose_factor_range_meta = args.dose_factor_range
        degradation_notes = "blur+downsample+noise; antialias disabled"
    elif args.degradation == 'blur':
        noise_sigma_range_meta = None  # ignored in 'blur'
        dose_factor_range_meta = None  # ignored in 'blur'
        degradation_notes = "blur+downsample only; noise ignored; antialias disabled"
    else:  # 'clean'
        noise_sigma_range_meta = None  # ignored in 'clean'
        dose_factor_range_meta = None  # ignored in 'clean'
        degradation_notes = f"downsample only; blur/noise ignored; antialias_clean={bool(args.antialias_clean)}"

    meta = {
        "experiment": exp_name,
        "run_dir": run_dir,
        "device": device_str,
        "amp_enabled": bool(amp_enabled),
        "optimizer": {"name": "Adam", "lr": args.lr, "betas": [0.9, 0.999]},
        "scale_factor": args.scale,
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "degradation": args.degradation,
        "blur_sigma_range": blur_sigma_range_eff,
        "blur_kernel": blur_kernel_eff,
        "noise_sigma_range_norm": noise_sigma_range_meta,
        "dose_factor_range": dose_factor_range_meta,
        "antialias_clean": bool(args.antialias_clean),
        "degradation_sampling": {
            "train": "volume (per-epoch resample)",
            "val": "volume (fixed per patient)",
            "test": "volume (fixed per patient)"
        },
        "splits": {
            "n_train_vols": len(train_dirs),
            "n_val_vols": len(val_dirs),
            "n_test_vols": len(test_dirs)
        },
        "notes": "L1 pretraining with per-epoch volume-wise degradation resampling",
        "degradation_notes": degradation_notes
    }
    # write JSON sidecar
    try:
        _ensure_dir(run_dir)
        with open(os.path.join(run_dir, "config.json"), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"[Meta] Wrote experiment metadata JSON -> {os.path.join(run_dir, 'config.json')}")
    except Exception as e:
        print(f"[Meta] Could not write metadata JSON: {e}")

    # Training with Early Stopping + Plot
    trained_model, train_losses, val_losses = train_sr_model(
        model, train_loader, val_loader,
        num_epochs=args.epochs, lr=args.lr, patience=args.patience,
        run_dir=run_dir,
        resume_path=args.resume,
        metadata=meta
    )

