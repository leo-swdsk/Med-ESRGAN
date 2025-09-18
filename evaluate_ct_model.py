import os
import csv
import json
import numpy as np
import argparse
import math
from collections import defaultdict

import torch
from datetime import datetime

from rrdb_ct_model import RRDBNet_CT
from seed_utils import fixed_seed_for_path
from ct_dataset_loader import CT_Dataset_SR
from ct_sr_evaluation import compare_methods
from window_presets import WINDOW_PRESETS


def _default_blur_sigma_range_for_scale(scale: int):
    if int(scale) == 2:
        return (0.8 - 0.1, 0.8 + 0.1)
    elif int(scale) == 4:
        return (1.2 - 0.15, 1.2 + 0.15)
    else:
        return (0.8 - 0.1, 0.8 + 0.1)

def get_patient_dirs(root_folder):
    dirs = [os.path.join(root_folder, d) for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
    return sorted(dirs)


def split_patients(root_folder, seed=42):
    patient_dirs = get_patient_dirs(root_folder)
    if len(patient_dirs) == 0:
        raise RuntimeError(f"No patient directories found under {root_folder}")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(patient_dirs))
    patient_dirs = [patient_dirs[i] for i in perm]
    n = len(patient_dirs)
    train_cut = int(0.70 * n)
    val_cut = int(0.85 * n)
    return {
        'train': patient_dirs[:train_cut],
        'val': patient_dirs[train_cut:val_cut],
        'test': patient_dirs[val_cut:]
    }


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def evaluate_split(root_folder, split_name, model_path, output_dir, device='cuda', scale=2,
                   hu_clip=(-1000, 2000), preset=None, window_center=None, window_width=None,
                   max_patients=None, max_slices_per_patient=None, slice_sampling='random', seed=42,
                   degradation='blurnoise', blur_sigma_range=None, blur_kernel=None,
                   noise_sigma_range_norm=(0.001, 0.003), dose_factor_range=(0.25, 0.5), antialias_clean=True, # antialias still disabled for blur and blurnoise
                   degradation_sampling='volume', deg_seed=42, deg_seed_mode='per_patient'):
    ensure_dir(output_dir)

    # Persist complete evaluation configuration for reproducibility
    eval_config = {
        'root_folder': root_folder,
        'split': split_name,
        'model_path': model_path,
        'output_dir': output_dir,
        'device': device,
        'scale': scale,
        'hu_clip': hu_clip,
        'metrics': {
            'mode': None,
            'preset': preset,
            'window_center': window_center,
            'window_width': window_width,
        },
        'max_patients': max_patients,
        'max_slices_per_patient': max_slices_per_patient,
        'slice_sampling': slice_sampling,
        'seed': seed,
        'degradation': degradation,
        'blur_sigma_range': blur_sigma_range,
        'blur_kernel': blur_kernel,
        'noise_sigma_range_norm': noise_sigma_range_norm,
        'dose_factor_range': dose_factor_range,
        'antialias_clean': antialias_clean,
        'degradation_sampling': degradation_sampling,
        'deg_seed': deg_seed,
        'deg_seed_mode': deg_seed_mode,
    }

    # Prepare model
    device_t = torch.device(device if (device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    model = RRDBNet_CT(scale=scale).to(device_t)
    state = torch.load(model_path, map_location=device_t)
    if isinstance(state, dict) and 'model' in state and all(k in state for k in ['epoch', 'model']):
        print("[Eval] Detected checkpoint dict; loading weights from 'model' key")
        state = state['model']
    model.load_state_dict(state)
    model.eval()

    # Determine patients for the requested split
    splits = split_patients(root_folder)
    patient_dirs = splits.get(split_name)
    if patient_dirs is None:
        raise ValueError(f"split_name must be one of train|val|test, got: {split_name}")
    if max_patients is not None:
        patient_dirs = patient_dirs[:max_patients]

    # Decide metrics mode and tag
    if isinstance(preset, str) and len(preset) > 0:
        metrics_mode = 'window'
        metrics_tag = f"preset-{preset}"
        # If WL/WW not provided, derive from preset
        if (window_center is None or window_width is None) and preset in WINDOW_PRESETS:
            try:
                window_center = float(WINDOW_PRESETS[preset]['center'])
                window_width = float(WINDOW_PRESETS[preset]['width'])
            except Exception:
                pass
    elif window_center is not None and window_width is not None:
        metrics_mode = 'window'
        metrics_tag = f"wlww_{int(window_center)}_{int(window_width)}"
    else:
        metrics_mode = 'global'
        try:
            lo, hi = float(hu_clip[0]), float(hu_clip[1])
            metrics_tag = f"globalHU_{int(lo)}_{int(hi)}"
        except Exception:
            metrics_tag = "globalHU"
    eval_config['metrics']['mode'] = metrics_mode
    eval_config['metrics']['window_center'] = window_center
    eval_config['metrics']['window_width'] = window_width

    # Output run directory and short artifact names
    model_tag = os.path.splitext(os.path.basename(model_path))[0]
    limited = (max_patients is not None) or (max_slices_per_patient is not None)
    suffix = f"_on_{'limited_' if limited else ''}{split_name}_set"
    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_tag = f"{split_name}__{model_tag}__{metrics_tag}{suffix}__{run_ts}"
    base_dir = os.path.join(output_dir, run_tag)
    ensure_dir(base_dir)
    print(f"[Eval] Run directory: {base_dir}")
    csv_path = os.path.join(base_dir, "metrics.csv")
    json_path = os.path.join(base_dir, "summary.json")

    # Collect per-slice metrics and per-patient aggregations
    fieldnames = ['patient_id', 'deg_seed', 'slice_index', 'method', 'MSE', 'RMSE', 'MAE', 'PSNR', 'SSIM', 'LPIPS', 'MA', 'NIQE', 'PI']
    rows = []
    patient_to_method_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    rng = np.random.default_rng(int(seed))

    if metrics_mode == 'global':
        print(f"[Eval] Metrics mode: global | hu_clip={hu_clip}")
    else:
        print(f"[Eval] Metrics mode: window | preset={preset} | WL/WW=({window_center},{window_width})")
    # Log requested/effective (range-level) degradation parameters
    _disp_blur_range = blur_sigma_range if blur_sigma_range is not None else _default_blur_sigma_range_for_scale(scale)
    _disp_kernel = blur_kernel if blur_kernel is not None else 'auto(by-sigma)'
    print(f"[Eval] Using degradation='{degradation}' | blur_sigma_range={_disp_blur_range} | blur_kernel={_disp_kernel} | noise_sigma_range_norm={noise_sigma_range_norm} | dose_factor_range={dose_factor_range} | antialias_clean={antialias_clean}")
    print(f"[Eval] Degradation sampling='{degradation_sampling}' | deg_seed_mode={deg_seed_mode} | base_seed={deg_seed}")

    # helper for per-patient deterministic seed
    patient_to_used_seed = {}
    with torch.no_grad():
        patient_to_degradation_sampled = {}
        for p_idx, patient_dir in enumerate(patient_dirs):
            patient_id = os.path.basename(patient_dir)

            # determine deterministic seed per patient (mode)
            used_seed = int(deg_seed) if str(deg_seed_mode) == 'global' else fixed_seed_for_path(patient_dir, int(deg_seed))
            patient_to_used_seed[patient_id] = used_seed
            print(f"[Eval] patient={patient_id} | used_deg_seed={used_seed}")

            # Full-slice dataset for evaluation (no random crop) with consistent degradation
            ds = CT_Dataset_SR(
                patient_dir,
                scale_factor=scale,
                do_random_crop=False,
                hu_clip=tuple(hu_clip),
                degradation=degradation,
                blur_sigma_range=blur_sigma_range,
                blur_kernel=blur_kernel,
                noise_sigma_range_norm=tuple(noise_sigma_range_norm),
                dose_factor_range=tuple(dose_factor_range),
                antialias_clean=antialias_clean,
                reverse_order=True,
                degradation_sampling=degradation_sampling,
                deg_seed=used_seed
            )
            # Record effective degradation configuration (ranges and sampled values if available)
            eff_blur_range = tuple(blur_sigma_range) if blur_sigma_range is not None else _default_blur_sigma_range_for_scale(scale)
            eval_config["blur_sigma_range_effective"] = [float(eff_blur_range[0]), float(eff_blur_range[1])]
            eval_config["blur_kernel_effective"] = blur_kernel
            eval_config["noise_sigma_range_norm_effective"] = [float(noise_sigma_range_norm[0]), float(noise_sigma_range_norm[1])]
            eval_config["dose_factor_range_effective"] = [float(dose_factor_range[0]), float(dose_factor_range[1])]
            if degradation_sampling == 'volume':
                try:
                    if hasattr(ds, 'deg_params') and ds.deg_params:
                        # Note: This field is overwritten for each patient inside the loop below and
                        # therefore ends up reflecting ONLY the values of the LAST processed patient in the split.
                        # It is NOT a global value "for all patients". For per-patient values, consult
                        # the separate 'patient_to_degradation_sampled' object stored in the summary JSON.
                        eval_config["degradation_sampled"] = ds.deg_params
                        if eval_config.get("blur_kernel_effective") is None:
                            eval_config["blur_kernel_effective"] = ds.deg_params.get("kernel")
                        # store per-patient sampled parameters for later viewer reuse
                        patient_to_degradation_sampled[patient_id] = ds.deg_params
                except Exception:
                    pass
            # Print effective parameters now that dataset is initialized
            print(f"[Eval] Effective degradation | blur_sigma_range={tuple(eval_config['blur_sigma_range_effective'])} | blur_kernel={eval_config['blur_kernel_effective']} | noise_sigma_range_norm={tuple(eval_config['noise_sigma_range_norm_effective'])} | dose_factor_range={tuple(eval_config['dose_factor_range_effective'])}")
            num_slices = len(ds)
            limit_slices = min(num_slices, max_slices_per_patient) if max_slices_per_patient else num_slices

            # choose slice indices according to sampling strategy
            if limit_slices >= num_slices:
                indices = list(range(num_slices))
            else:
                if slice_sampling == 'first':
                    indices = list(range(limit_slices))
                else:  # 'random'
                    indices = list(rng.choice(num_slices, size=limit_slices, replace=False))
                    indices.sort()

            for s_idx in indices:
                lr, hr = ds[s_idx]
                # Debug identity print to verify consistent slice matching
                try:
                    import pydicom
                    # read metadata from dataset path since CT_Dataset_SR stores file ordering
                    # by construction, ds.paths is aligned to ds indices
                    dicom_path = ds.paths[s_idx]
                    # dsi = pydicom.dcmread(dicom_path, stop_before_pixels=True, force=True)
                    # inst = getattr(dsi, 'InstanceNumber', None)
                    # uid = str(getattr(dsi, 'SOPInstanceUID', ''))
                    if (s_idx % 20) == 0:
                        print(f"[Eval SliceMeta] patient={patient_id} idx={s_idx} Path={dicom_path}")
                except Exception:
                    pass
                results = compare_methods(
                    lr, hr, model,
                    hu_clip=hu_clip,
                    metrics_mode=metrics_mode,
                    window_center=window_center,
                    window_width=window_width,
                    metrics_device=('cuda' if (device_t.type == 'cuda' and torch.cuda.is_available()) else 'cpu')
                )

                for method_name, metrics in results.items():
                    rows.append({
                        'patient_id': patient_id,
                        'deg_seed': patient_to_used_seed.get(patient_id, None),
                        'slice_index': s_idx,
                        'method': method_name,
                        'MSE': float(metrics['MSE']),
                        'RMSE': float(metrics['RMSE']),
                        'MAE': float(metrics['MAE']),
                        'PSNR': float(metrics['PSNR']),
                        'SSIM': float(metrics['SSIM']),
                        'LPIPS': float(metrics.get('LPIPS', float('nan'))),
                        'MA': float(metrics.get('MA', float('nan'))),
                        'NIQE': float(metrics.get('NIQE', float('nan'))),
                        'PI': float(metrics.get('PI', float('nan')))
                    })
                    for metric_name, metric_value in metrics.items():
                        patient_to_method_metrics[patient_id][method_name][metric_name].append(float(metric_value))

    # Write per-slice CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Aggregate statistics
    def mean_std(values):
        finite_vals = [v for v in values if math.isfinite(v)]
        if len(finite_vals) == 0:
            return 0.0, 0.0
        m = sum(finite_vals) / len(finite_vals)
        var = sum((v - m) ** 2 for v in finite_vals) / max(1, (len(finite_vals) - 1))
        return m, var ** 0.5

    # Per-slice global aggregation
    global_by_method = defaultdict(lambda: defaultdict(list))
    for r in rows:
        method_name = r['method']
        for metric_name in ['MSE', 'RMSE', 'MAE', 'PSNR', 'SSIM', 'LPIPS', 'MA', 'NIQE', 'PI']:
            global_by_method[method_name][metric_name].append(float(r[metric_name]))

    global_summary = {}
    for method_name, metric_map in global_by_method.items():
        global_summary[method_name] = {}
        for metric_name, values in metric_map.items():
            m, s = mean_std(values)
            global_summary[method_name][metric_name] = {'mean': m, 'std': s, 'n': len(values)}

    # Per-patient aggregation (each patient contributes one mean per metric)
    per_patient_summary = {}
    for patient_id, method_map in patient_to_method_metrics.items():
        per_patient_summary[patient_id] = {}
        for method_name, metric_map in method_map.items():
            per_patient_summary[patient_id][method_name] = {}
            for metric_name, values in metric_map.items():
                m, s = mean_std(values)
                per_patient_summary[patient_id][method_name][metric_name] = {'mean': m, 'std': s, 'n': len(values)}

    # Aggregate across patients (each patient weighted equally)
    patient_level_agg = {}
    methods = set()
    for patient_id in per_patient_summary:
        methods.update(per_patient_summary[patient_id].keys())
    for method_name in methods:
        patient_level_agg[method_name] = {}
        for metric_name in ['MSE', 'RMSE', 'MAE', 'PSNR', 'SSIM', 'LPIPS', 'MA', 'NIQE', 'PI']:
            patient_means = []
            for patient_id in per_patient_summary:
                if method_name in per_patient_summary[patient_id]:
                    if metric_name in per_patient_summary[patient_id][method_name]:
                        patient_means.append(per_patient_summary[patient_id][method_name][metric_name]['mean'])
            m, s = mean_std(patient_means)
            patient_level_agg[method_name][metric_name] = {'mean_of_patient_means': m, 'std_of_patient_means': s, 'num_patients': len(patient_means)}

    # Before writing JSON: ensure config contains effective (non-null) values for blur params
    try:
        if eval_config.get('blur_sigma_range') is None and eval_config.get('blur_sigma_range_effective') is not None:
            # store the effective range actually used when None was provided
            eval_config['blur_sigma_range'] = eval_config['blur_sigma_range_effective']
        if eval_config.get('blur_kernel') is None:
            if eval_config.get('blur_kernel_effective') is not None:
                eval_config['blur_kernel'] = eval_config['blur_kernel_effective']
            else:
                eval_config['blur_kernel'] = 'auto(by-sigma)'
    except Exception:
        pass

    # Write JSON summary
    summary = {
        'split': split_name,
        'num_patients': len(patient_dirs),
        'paths': {
            'csv': 'metrics.csv',
            'json': 'summary.json',
            'plots_dir': 'plots/'
        },
        'config': eval_config,
        'deg_seeds_per_patient': patient_to_used_seed,
        'patient_to_degradation_sampled': patient_to_degradation_sampled,
        'global_per_slice': global_summary,
        'per_patient': per_patient_summary,
        'patient_level_aggregate': patient_level_agg
    }
    # Sanitize JSON (replace inf/nan with strings) for strict JSON compatibility
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        if isinstance(obj, float):
            if math.isfinite(obj):
                return obj
            if math.isinf(obj):
                return 'inf'
            return 'nan'
        return obj

    with open(json_path, 'w') as f:
        json.dump(sanitize(summary), f, indent=2, allow_nan=False)

    print(f"[Eval] Wrote per-slice CSV to: {csv_path}")
    print(f"[Eval] Wrote summary JSON to: {json_path}")

    # -------------------- Headless plot routine --------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns  # optional, for nicer palettes/heatmaps
        except Exception:
            sns = None

        plots_dir = os.path.join(base_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Common config
        def _get_palette(n):
            if sns is not None:
                return sns.color_palette('Set2', n_colors=n)
            # simple matplotlib tab10 fallback
            import itertools
            colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
            return list(itertools.islice(colors, n))

        # Consistent method order across plots
        present_methods = list(global_summary.keys())
        preferred = [
            'Interpolation (Bilinear)',
            'Interpolation (Bicubic)',
            'Model (RRDB)'
        ]
        method_order = [m for m in preferred if m in present_methods] + [m for m in sorted(present_methods) if m not in preferred]

        # Helper to filter finite values
        def _finite(vals):
            return [v for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]

        # 1) Bar charts – Global (per slice)
        metrics_to_plot = [
            ('PSNR', 'PSNR (↑)'),
            ('SSIM', 'SSIM (↑)'),
            ('LPIPS', 'LPIPS (↓)'),
            ('PI', 'PI (↓)'),
            ('RMSE', 'RMSE (↓)'),
            ('MAE', 'MAE (↓)')
        ]

        for metric_key, metric_title in metrics_to_plot:
            xs, ys, es, used_methods = [], [], [], []
            for m in method_order:
                g = global_summary.get(m, {}).get(metric_key)
                if g is None or not all(k in g for k in ['mean','std']):
                    print(f"[Plots] Skipping method '{m}' for global {metric_key}: no stats")
                    continue
                if not (math.isfinite(g['mean']) and math.isfinite(g['std'])):
                    print(f"[Plots] Skipping method '{m}' for global {metric_key}: non-finite stats")
                    continue
                used_methods.append(m)
                xs.append(m)
                ys.append(float(g['mean']))
                es.append(float(g['std']))
            if len(used_methods) == 0:
                print(f"[Plots] No valid data for global bar chart: {metric_key}")
            else:
                plt.figure(figsize=(8, 5))
                colors = _get_palette(len(used_methods))
                bars = plt.bar(range(len(xs)), ys, yerr=es, color=colors, capsize=4)
                plt.xticks(range(len(xs)), xs, rotation=20, ha='right')
                plt.ylabel(metric_title)
                plt.xlabel('Method')
                plt.title(f"Global per-slice: {metric_title}")
                plt.tight_layout()
                plot_id = f"bar_global_{metric_key}"
                out_path = os.path.join(plots_dir, f"{plot_id}.png")
                plt.savefig(out_path, dpi=150)
                plt.close()
                print(f"[Plots] Saved: {out_path}")

        # 2) Bar charts – Patient-Level (mean of patient means)
        for metric_key, metric_title in metrics_to_plot:
            xs, ys, es, used_methods = [], [], [], []
            for m in method_order:
                g = patient_level_agg.get(m, {}).get(metric_key)
                if g is None or not all(k in g for k in ['mean_of_patient_means','std_of_patient_means']):
                    print(f"[Plots] Skipping method '{m}' for patient-level {metric_key}: no stats")
                    continue
                mean_v = float(g['mean_of_patient_means'])
                std_v = float(g['std_of_patient_means'])
                if not (math.isfinite(mean_v) and math.isfinite(std_v)):
                    print(f"[Plots] Skipping method '{m}' for patient-level {metric_key}: non-finite stats")
                    continue
                used_methods.append(m)
                xs.append(m)
                ys.append(mean_v)
                es.append(std_v)
            if len(used_methods) == 0:
                print(f"[Plots] No valid data for patient-level bar chart: {metric_key}")
            else:
                plt.figure(figsize=(8, 5))
                colors = _get_palette(len(used_methods))
                bars = plt.bar(range(len(xs)), ys, yerr=es, color=colors, capsize=4)
                plt.xticks(range(len(xs)), xs, rotation=20, ha='right')
                plt.ylabel(metric_title)
                plt.xlabel('Method')
                plt.title(f"Patient-level: {metric_title}")
                plt.tight_layout()
                plot_id = f"bar_patient_{metric_key}"
                out_path = os.path.join(plots_dir, f"{plot_id}.png")
                plt.savefig(out_path, dpi=150)
                plt.close()
                print(f"[Plots] Saved: {out_path}")

        # 3) Boxplots – Distributions (per slice) for PSNR, SSIM, LPIPS
        box_metrics = [('PSNR', 'PSNR (↑)'), ('SSIM', 'SSIM (↑)'), ('LPIPS', 'LPIPS (↓)')]
        # Collect data per method from rows
        for metric_key, metric_title in box_metrics:
            data = []
            labels = []
            for m in method_order:
                vals = [float(r[metric_key]) for r in rows if r.get('method') == m and (metric_key in r)]
                vals = _finite(vals)
                if len(vals) == 0:
                    print(f"[Plots] Skipping method '{m}' in boxplot {metric_key}: no finite values")
                    continue
                labels.append(m)
                data.append(vals)
            if len(data) == 0:
                print(f"[Plots] No valid data for boxplot: {metric_key}")
            else:
                plt.figure(figsize=(8, 5))
                if sns is not None:
                    # Build a long-form dataset for seaborn
                    import pandas as pd
                    long_vals = []
                    long_meth = []
                    for lbl, arr in zip(labels, data):
                        long_vals.extend(arr)
                        long_meth.extend([lbl] * len(arr))
                    df = pd.DataFrame({'Method': long_meth, metric_key: long_vals})
                    ax = sns.boxplot(data=df, x='Method', y=metric_key, order=labels, hue='Method', palette=_get_palette(len(labels)), dodge=False)
                    if ax.legend_ is not None:
                        ax.legend_.remove()
                    import matplotlib.pyplot as _plt
                    _plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
                    plt.ylabel(metric_title)
                    plt.xlabel('Method')
                    plt.title(f"Distribution per slice: {metric_title}")
                else:
                    plt.boxplot(data, tick_labels=labels, patch_artist=True)
                    plt.xticks(rotation=20, ha='right')
                    plt.ylabel(metric_title)
                    plt.xlabel('Method')
                    plt.title(f"Distribution per slice: {metric_title}")
                plt.tight_layout()
                plot_id = f"boxplot_{metric_key}"
                out_path = os.path.join(plots_dir, f"{plot_id}.png")
                plt.savefig(out_path, dpi=150)
                plt.close()
                print(f"[Plots] Saved: {out_path}")

        # 4) Correlation heatmap of metrics for "Model (RRDB)" (fallback to first available)
        corr_metrics = ['PSNR', 'SSIM', 'LPIPS', 'PI', 'RMSE', 'MAE']
        method_for_corr = 'Model (RRDB)' if 'Model (RRDB)' in method_order else (method_order[0] if len(method_order) > 0 else None)
        if method_for_corr is None:
            print("[Plots] Skipping correlation heatmap: no methods available")
        else:
            # Gather per-slice rows for selected method, ensuring all metrics are finite per row
            matrix = []
            for r in rows:
                if r.get('method') != method_for_corr:
                    continue
                vals = []
                ok = True
                for mk in corr_metrics:
                    v = float(r.get(mk, float('nan')))
                    if not math.isfinite(v):
                        ok = False
                        break
                    vals.append(v)
                if ok:
                    matrix.append(vals)
            if len(matrix) < 2:
                print(f"[Plots] Skipping correlation heatmap: insufficient valid samples for method '{method_for_corr}'")
            else:
                import numpy as _np
                mat = _np.asarray(matrix, dtype=float)
                # Compute Pearson correlation across columns
                with _np.errstate(invalid='ignore'):
                    C = _np.corrcoef(mat, rowvar=False)
                # Replace any nan with 0 for visualization simplicity
                C = _np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
                plt.figure(figsize=(7, 6))
                if sns is not None:
                    ax = sns.heatmap(C, xticklabels=corr_metrics, yticklabels=corr_metrics, annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1, square=True)
                    plt.title(f"Pearson correlation – {method_for_corr}")
                else:
                    plt.imshow(C, cmap='coolwarm', vmin=-1, vmax=1)
                    plt.colorbar()
                    plt.xticks(range(len(corr_metrics)), corr_metrics, rotation=20, ha='right')
                    plt.yticks(range(len(corr_metrics)), corr_metrics)
                    plt.title(f"Pearson correlation – {method_for_corr}")
                plt.tight_layout()
                def _sanitize_method_for_fn(name: str) -> str:
                    s = name.lower()
                    for ch in [' ', '(', ')', '[', ']', '{', '}', '/', '\\', ':', ';', ',', '\'', '"']:
                        s = s.replace(ch, '-')
                    s = ''.join(c for c in s if (c.isalnum() or c in ['-','_']))
                    while '--' in s:
                        s = s.replace('--', '-')
                    return s.strip('-_')
                plot_id = "corr_heatmap" if method_for_corr == 'Model (RRDB)' else f"corr_heatmap_{_sanitize_method_for_fn(method_for_corr)}"
                out_path = os.path.join(plots_dir, f"{plot_id}.png")
                plt.savefig(out_path, dpi=150)
                plt.close()
                print(f"[Plots] Saved: {out_path}")
    except Exception as e:
        print(f"[Plots] Plotting failed: {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate CT SR model with aggregated metrics and summaries')
    parser.add_argument('--root', type=str, required=True, help='Root folder containing patient subfolders (same as training root)')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help='Which split to evaluate')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--output_dir', type=str, default='eval_results', help='Directory to write CSV/JSON outputs')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--scale', type=int, default=2, help='Upsampling scale (must match model)')
    # Metrics windowing (degradation always global HU clip [-1000,2000])
    parser.add_argument('--hu_clip', type=float, nargs=2, default=[-1000.0, 2000.0], help='HU clip range for degradation/global denorm [lo hi]')
    parser.add_argument('--preset', type=str, default=None, help='Window preset name for metrics (overrides WL/WW if set)')
    parser.add_argument('--window_center', type=float, default=None, help='Window center for metrics (used if preset not set)')
    parser.add_argument('--window_width', type=float, default=None, help='Window width for metrics (used if preset not set)')
    parser.add_argument('--max_patients', type=int, default=None, help='Optional limit of patients for a quick run')
    parser.add_argument('--max_slices_per_patient', type=int, default=None, help='Optional limit of slices per patient for a quick run')
    parser.add_argument('--slice_sampling', type=str, default='random', choices=['first', 'random'], help='How to select slices when limited')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible sampling')
    # Degradation flags (default to blurnoise)
    parser.add_argument('--degradation', type=str, default='blurnoise', choices=['clean', 'blur', 'blurnoise'], help='Degradation pipeline for LR generation (default: blurnoise)')
    parser.add_argument('--blur_sigma_range', type=float, nargs=2, default=None, help='Range [lo hi] of Gaussian blur sigma; if None, defaults by scale')
    parser.add_argument('--blur_kernel', type=int, default=None, help='Explicit odd kernel size; if None, derived from sigma')
    parser.add_argument('--noise_sigma_range_norm', type=float, nargs=2, default=[0.001, 0.003], help='Gaussian noise sigma range on normalized [-1,1] image')
    parser.add_argument('--dose_factor_range', type=float, nargs=2, default=[0.25, 0.5], help='Dose factor range; noise scales ~ 1/sqrt(dose)')
    parser.add_argument('--antialias_clean', action='store_true', help='Use antialias in clean downsample')
    parser.add_argument('--degradation_sampling', type=str, default='volume', choices=['volume','slice','det-slice'], help='Degradation sampling mode (volume|slice|det-slice)')
    parser.add_argument('--deg_seed', type=int, default=42, help='Seed for degradation sampling (volume or det-slice)')
    parser.add_argument('--deg_seed_mode', type=str, default='per_patient', choices=['global','per_patient'], help='How to derive degradation seed per patient (global same for all, per_patient hashed per path)')
    args = parser.parse_args()

    # Echo CLI config
    print("[Eval-Args] root=", args.root)
    print("[Eval-Args] split=", args.split)
    print("[Eval-Args] model_path=", args.model_path)
    print("[Eval-Args] output_dir=", args.output_dir)
    print("[Eval-Args] device=", args.device)
    print("[Eval-Args] scale=", args.scale)
    print("[Eval-Args] hu_clip=", args.hu_clip)
    print("[Eval-Args] preset=", args.preset)
    print("[Eval-Args] window_center=", args.window_center)
    print("[Eval-Args] window_width=", args.window_width)
    print("[Eval-Args] max_patients=", args.max_patients)
    print("[Eval-Args] max_slices_per_patient=", args.max_slices_per_patient)
    print("[Eval-Args] slice_sampling=", args.slice_sampling)
    print("[Eval-Args] seed=", args.seed)
    print("[Eval-Args] degradation=", args.degradation)
    # Display effective defaults for blur params if not provided
    _disp_blur_sigma_range_cli = args.blur_sigma_range if args.blur_sigma_range is not None else _default_blur_sigma_range_for_scale(args.scale)
    _disp_blur_kernel_cli = args.blur_kernel if args.blur_kernel is not None else 'auto(by-sigma)'
    print("[Eval-Args] blur_sigma_range=", _disp_blur_sigma_range_cli)
    print("[Eval-Args] blur_kernel=", _disp_blur_kernel_cli)
    print("[Eval-Args] noise_sigma_range_norm=", args.noise_sigma_range_norm)
    print("[Eval-Args] dose_factor_range=", args.dose_factor_range)
    print("[Eval-Args] antialias_clean=", args.antialias_clean)
    print("[Eval-Args] degradation_sampling=", args.degradation_sampling)
    print("[Eval-Args] deg_seed=", args.deg_seed)

    evaluate_split(
        root_folder=args.root,
        split_name=args.split,
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        scale=args.scale,
        hu_clip=tuple(args.hu_clip),
        preset=args.preset,
        window_center=args.window_center,
        window_width=args.window_width,
        max_patients=args.max_patients,
        max_slices_per_patient=args.max_slices_per_patient,
        slice_sampling=args.slice_sampling,
        seed=args.seed,
        degradation=args.degradation,
        blur_sigma_range=args.blur_sigma_range,
        blur_kernel=args.blur_kernel,
        noise_sigma_range_norm=args.noise_sigma_range_norm,
        dose_factor_range=args.dose_factor_range,
        antialias_clean=args.antialias_clean
        ,degradation_sampling=args.degradation_sampling
        ,deg_seed=args.deg_seed
        ,deg_seed_mode=args.deg_seed_mode
    )


if __name__ == '__main__':
    main()


