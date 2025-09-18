import os
import json
import argparse
import numpy as np
import platform
from datetime import datetime, timezone
import torch
import pydicom

from evaluate_ct_model import split_patients
from ct_dataset_loader import is_ct_image_dicom

try:
    UTC = datetime.UTC          # Python 3.11+
except AttributeError:
    UTC = timezone.utc          # older Python versions


def count_ct_slices_silent(patient_dir):
    count = 0
    ignore_dirs = {'ctkDICOM-Database', '.git', '__pycache__'}
    for root, dirs, files in os.walk(patient_dir):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for f in files:
            if f.lower().endswith('.dcm'):
                path = os.path.join(root, f)
                if is_ct_image_dicom(path):
                    count += 1
    return count


def detect_patient_device_and_geometry(patient_dir):
    """Return (modality, device_name, rows, cols, slice_thickness_mm, pixel_spacing_mm)
    using header-only read from the first CT Image DICOM found. Falls back to first DICOM if no CT found.
    pixel_spacing_mm is a two-element list [row_mm, col_mm] when available.
    """
    ignore_dirs = {'ctkDICOM-Database', '.git', '__pycache__'}

    def read_fields(path):
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        modality = str(getattr(ds, 'Modality', ''))
        manufacturer = str(getattr(ds, 'Manufacturer', '')).strip()
        model_name = str(getattr(ds, 'ManufacturerModelName', '')).strip()
        if manufacturer and model_name:
            device = f"{manufacturer} {model_name}"
        else:
            device = manufacturer or model_name
        rows = int(getattr(ds, 'Rows', 0)) if hasattr(ds, 'Rows') else 0
        cols = int(getattr(ds, 'Columns', 0)) if hasattr(ds, 'Columns') else 0
        slice_thickness = None
        try:
            st = getattr(ds, 'SliceThickness', None)
            if st is not None:
                slice_thickness = float(st)
        except Exception:
            slice_thickness = None
        pixel_spacing = None
        try:
            ps = getattr(ds, 'PixelSpacing', None)
            if ps is not None and len(ps) >= 2:
                pixel_spacing = [float(ps[0]), float(ps[1])]
        except Exception:
            pixel_spacing = None
        return modality, device, rows, cols, slice_thickness, pixel_spacing

    # First try to find a CT image DICOM
    for root, dirs, files in os.walk(patient_dir):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for f in files:
            if f.lower().endswith('.dcm'):
                path = os.path.join(root, f)
                try:
                    if is_ct_image_dicom(path):
                        return read_fields(path)
                except Exception:
                    continue

    # Fallback: first available DICOM
    for root, dirs, files in os.walk(patient_dir):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for f in files:
            if f.lower().endswith('.dcm'):
                path = os.path.join(root, f)
                try:
                    return read_fields(path)
                except Exception:
                    continue

    return '', '', 0, 0, None, None


def dump_split(root_folder, seed=42, output_path=None):
    # Determine device info
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()

    # Build split deterministically
    splits = split_patients(root_folder, seed=seed)

    # Compose payload
    payload = {
        'meta': {
            'timestamp': datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
            'root': root_folder,
            'seed': seed,
            'device': device,
            'device_name': device_name,
            'python': platform.python_version(),
            'platform': platform.platform(),
        },
        'counts': {},
        'splits': {k: [] for k in ['train', 'val', 'test']}
    }

    # Fill splits with patient info and counts
    for split_name, patients in splits.items():
        for p in patients:
            patient_id = os.path.basename(p)
            num_slices = count_ct_slices_silent(p)
            modality, device_name, rows, cols, slice_thickness, pixel_spacing = detect_patient_device_and_geometry(p)
            payload['splits'][split_name].append({
                'patient_id': patient_id,
                'path': p,
                'num_slices': num_slices,
                'modality': modality,
                'device': device_name,
                'rows': rows,
                'cols': cols,
                'slice_thickness_mm': slice_thickness,
                'pixel_spacing_mm': pixel_spacing
            })

    # Aggregate counts per split
    for split_name in ['train', 'val', 'test']:
        entries = payload['splits'][split_name]
        payload['counts'][split_name] = {
            'num_patients': len(entries),
            'total_slices': sum(e['num_slices'] for e in entries)
        }

    # Default output path
    if output_path is None:
        os.makedirs('splits', exist_ok=True)
        output_path = os.path.join('splits', f'patient_split_seed{seed}.json')

    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"[Split] Wrote split mapping to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Dump deterministic patient split with per-patient slice counts and metadata')
    parser.add_argument('--root', type=str, required=True, help='Root folder containing patient subfolders')
    parser.add_argument('--seed', type=int, default=42, help='Random seed used for splitting (must match training/eval)')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path (default: splits/patient_split_seed<seed>.json)')
    args = parser.parse_args()

    dump_split(args.root, seed=args.seed, output_path=args.output)


if __name__ == '__main__':
    main()


