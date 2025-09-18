import os
import argparse
import csv
import json
from typing import List, Tuple, Optional

import numpy as np
import pydicom
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from pydicom.tag import Tag
from pydicom.dataelem import DataElement
try:
    from pydicom.uid import PYDICOM_IMPLEMENTATION_UID, PYDICOM_IMPLEMENTATION_VERSION
except Exception:
    PYDICOM_IMPLEMENTATION_UID = generate_uid()
    PYDICOM_IMPLEMENTATION_VERSION = 'PYDICOM'

try:
    from scipy.ndimage import zoom as nd_zoom
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from ct_dataset_loader import is_ct_image_dicom, find_dicom_files_recursively


def get_patient_dirs(root_folder: str) -> List[str]:
    dirs = [os.path.join(root_folder, d) for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
    return sorted(dirs)


def read_spacing_and_geometry(ds: pydicom.dataset.FileDataset) -> Tuple[Optional[float], Optional[float], Optional[float], int, int]:
    row_spacing = None
    col_spacing = None
    slice_thickness = None
    try:
        ps = getattr(ds, 'PixelSpacing', None)
        if ps is not None and len(ps) >= 2:
            row_spacing = float(ps[0])
            col_spacing = float(ps[1])
    except Exception:
        row_spacing = None
        col_spacing = None
    try:
        st = getattr(ds, 'SliceThickness', None)
        if st is not None:
            slice_thickness = float(st)
    except Exception:
        slice_thickness = None
    rows = int(getattr(ds, 'Rows', 0)) if hasattr(ds, 'Rows') else 0
    cols = int(getattr(ds, 'Columns', 0)) if hasattr(ds, 'Columns') else 0
    return row_spacing, col_spacing, slice_thickness, rows, cols


def resample_slice(arr: np.ndarray, src_row_mm: float, src_col_mm: float, target_mm: float) -> np.ndarray:
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for resampling (scipy.ndimage.zoom not available)")
    # zoom factors >1 means upsample
    zoom_r = float(src_row_mm) / float(target_mm)
    zoom_c = float(src_col_mm) / float(target_mm)
    # order=1 -> linear interpolation
    out = nd_zoom(arr, zoom=(zoom_r, zoom_c), order=1, prefilter=False)
    return out


def ensure_out_path(out_root: str, patient_id: str, src_patient_dir: str, src_file_path: str) -> str:
    rel_inside_patient = os.path.relpath(os.path.dirname(src_file_path), src_patient_dir)
    # Suffix patient folder with 'pp' to mark preprocessing
    patient_folder = patient_id if patient_id.endswith('pp') else f"{patient_id}pp"
    dst_dir = os.path.join(out_root, patient_folder, rel_inside_patient)
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, os.path.basename(src_file_path))
    return dst_path


def write_resampled_dicom(ds: pydicom.dataset.FileDataset, resampled: np.ndarray, target_mm: float, out_path: str):
    # Cast back to original dtype with safe clipping
    orig_dtype = ds.pixel_array.dtype
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        resampled = np.clip(np.rint(resampled), info.min, info.max).astype(orig_dtype)
    else:
        resampled = resampled.astype(orig_dtype)

    ds.Rows = int(resampled.shape[0])
    ds.Columns = int(resampled.shape[1])
    ds.PixelSpacing = [float(target_mm), float(target_mm)]

    # Ensure CT image identity (avoid accidental SEG headers, ensure scalar volume compliance)
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
    ds.Modality = 'CT'
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'

    # Ensure pixel type fields consistent with dtype
    bits_alloc = resampled.dtype.itemsize * 8
    ds.BitsAllocated = bits_alloc
    ds.BitsStored = bits_alloc
    ds.HighBit = bits_alloc - 1

    # Update meta to uncompressed explicit VR little endian
    if not hasattr(ds, 'file_meta') or ds.file_meta is None:
        from pydicom.dataset import FileMetaDataset
        ds.file_meta = FileMetaDataset()
    # File Meta Information (Part 10) – required fields
    ds.file_meta.FileMetaInformationVersion = b"\x00\x01"
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    # Generate a new SOPInstanceUID to mark derived image (keep Study/Series)
    try:
        ds.SOPInstanceUID = generate_uid()
    except Exception:
        pass
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
    ds.file_meta.ImplementationVersionName = str(PYDICOM_IMPLEMENTATION_VERSION)[:16]
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # Replace pixel data
    ds.PixelData = resampled.tobytes()

    # Ensure required CT image attributes exist to satisfy strict readers
    if not hasattr(ds, 'RescaleSlope'):
        ds.RescaleSlope = 1
    if not hasattr(ds, 'RescaleIntercept'):
        ds.RescaleIntercept = 0
    if not hasattr(ds, 'ImageType'):
        ds.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
    if not hasattr(ds, 'PatientID'):
        ds.PatientID = 'Anonymous'
    if not hasattr(ds, 'PatientName'):
        ds.PatientName = 'Anonymous'

    # Optional but good practice: update Largest/SmallestImagePixelValue with explicit VR resolution
    try:
        # Ensure PixelRepresentation is known (0=unsigned, 1=signed)
        pixel_rep = getattr(ds, 'PixelRepresentation', None)
        if pixel_rep not in (0, 1):
            # Infer from dtype if missing
            pixel_rep = 1 if np.issubdtype(resampled.dtype, np.signedinteger) else 0
            ds.PixelRepresentation = pixel_rep

        vr = 'SS' if pixel_rep == 1 else 'US'
        min_val = int(resampled.min())
        max_val = int(resampled.max())
        ds[Tag(0x0028, 0x0106)] = DataElement(Tag(0x0028, 0x0106), vr, min_val)
        ds[Tag(0x0028, 0x0107)] = DataElement(Tag(0x0028, 0x0107), vr, max_val)
    except Exception:
        # If anything goes wrong, remove these optional attributes to avoid write errors
        try:
            if Tag(0x0028, 0x0106) in ds:
                del ds[Tag(0x0028, 0x0106)]
            if Tag(0x0028, 0x0107) in ds:
                del ds[Tag(0x0028, 0x0107)]
        except Exception:
            pass

    ds.save_as(out_path, write_like_original=False)


def process_patient(patient_dir: str, out_root: str, target_mm: float) -> Optional[dict]:
    patient_id = os.path.basename(patient_dir.rstrip(os.sep))
    dicom_files = find_dicom_files_recursively(patient_dir)
    if len(dicom_files) == 0:
        print(f"[Preproc] {patient_id}: no CT DICOMs found, skipping")
        return None

    # Determine reference spacing/geometry from the first readable CT image
    first_row_mm = None
    first_col_mm = None
    first_thickness = None
    first_rows = None
    first_cols = None
    first_new_rows = None
    first_new_cols = None

    processed = 0
    for i, path in enumerate(dicom_files):
        try:
            ds = pydicom.dcmread(path, force=True)
        except Exception:
            continue

        try:
            if ds.SamplesPerPixel != 1:
                # Skip non-mono images
                continue
        except Exception:
            pass

        try:
            arr = ds.pixel_array
        except Exception:
            # Skip non-image or unsupported pixel data files (e.g., SEG objects)
            continue

        row_mm, col_mm, slice_thickness, rows, cols = read_spacing_and_geometry(ds)
        if row_mm is None or col_mm is None:
            # Cannot resample without spacing
            print(f"[Preproc] {patient_id}: missing PixelSpacing for file {os.path.basename(path)}, skipping slice")
            continue

        # Compute resampled slice
        res = resample_slice(arr, row_mm, col_mm, target_mm)

        # Record reference info (from first valid slice)
        if first_row_mm is None:
            first_row_mm, first_col_mm = row_mm, col_mm
            first_thickness = slice_thickness
            first_rows, first_cols = rows, cols
            first_new_rows, first_new_cols = int(res.shape[0]), int(res.shape[1])

        # Build destination path mirroring the patient sub-structure
        dst_path = ensure_out_path(out_root, patient_id, patient_dir, path)
        try:
            write_resampled_dicom(ds, res, target_mm, dst_path)
            processed += 1
        except Exception as e:
            print(f"[Preproc] {patient_id}: failed to write {os.path.basename(path)} -> {e}")
            continue

    if processed == 0:
        print(f"[Preproc] {patient_id}: no slices written (all skipped)")
        return None

    return {
        'patient_id': patient_id,
        'num_slices': processed,
        'orig_row_mm': first_row_mm,
        'orig_col_mm': first_col_mm,
        'slice_thickness_mm': first_thickness,
        'new_row_mm': target_mm,
        'new_col_mm': target_mm,
        'orig_rows': first_rows,
        'orig_cols': first_cols,
        'new_rows': first_new_rows,
        'new_cols': first_new_cols,
    }


def write_log(csv_path: str, rows: List[dict]):
    header = [
        'patient_id', 'num_slices',
        'orig_row_mm', 'orig_col_mm', 'slice_thickness_mm',
        'new_row_mm', 'new_col_mm',
        'orig_rows', 'orig_cols', 'new_rows', 'new_cols'
    ]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json_log(json_path: str, rows: List[dict]):
    # Convert CSV-style rows into JSON with combined pixel spacing
    json_rows = []
    for r in rows:
        entry = {
            'patient_id': r.get('patient_id'),
            'num_slices': r.get('num_slices'),
            'pixel_spacing_mm': [r.get('orig_row_mm'), r.get('orig_col_mm')],
            'slice_thickness_mm': r.get('slice_thickness_mm'),
            'new_pixel_spacing_mm': [r.get('new_row_mm'), r.get('new_col_mm')],
            'orig_rows': r.get('orig_rows'),
            'orig_cols': r.get('orig_cols'),
            'new_rows': r.get('new_rows'),
            'new_cols': r.get('new_cols'),
        }
        json_rows.append(entry)
    with open(json_path, 'w') as f:
        json.dump(json_rows, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Resample CT DICOM slices to uniform in-plane spacing and persist results')
    parser.add_argument('--root', type=str, required=True, help='Root folder with per-patient subfolders')
    parser.add_argument('--out_dir', type=str, default=None, help='Output root for resampled data (default: ESRGAN/preprocessed_data)')
    parser.add_argument('--target_spacing', type=float, default=0.8, help='Target in-plane spacing in mm (row,col)')
    args = parser.parse_args()

    if not _HAS_SCIPY:
        raise SystemExit('scipy is required but not available. Please install scipy to run this script.')

    # Determine output root (default: next to this script under ESRGAN/)
    if args.out_dir is not None:
        out_root = args.out_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_root = os.path.join(script_dir, 'preprocessed_data')
    os.makedirs(out_root, exist_ok=True)
    log_rows: List[dict] = []

    # Build patient list and exclude the output folder if it sits under root
    patients = get_patient_dirs(args.root)
    if len(patients) == 0:
        raise RuntimeError(f"No patient directories found under {args.root}")
    out_base = os.path.basename(os.path.normpath(out_root))
    patients = [p for p in patients if os.path.basename(p) != out_base]

    print(f"[Preproc] Starting resample to {args.target_spacing} mm for {len(patients)} patients")
    for p_dir in patients:
        pid = os.path.basename(p_dir)
        print(f"[Preproc] Processing patient: {pid}")
        row = process_patient(p_dir, out_root, args.target_spacing)
        if row is not None:
            log_rows.append(row)

    # Write consolidated logs in output folder only
    csv_path_out = os.path.join(out_root, 'preprocessing_log.csv')
    write_log(csv_path_out, log_rows)
    json_path_out = os.path.join(out_root, 'preprocessing_log.json')
    write_json_log(json_path_out, log_rows)
    print(f"[Preproc] Wrote logs: {csv_path_out}, {json_path_out}")
    print(f"[Preproc] Done. Output root: {os.path.abspath(out_root)}")


if __name__ == '__main__':
    main()


