import os
import numpy as np
import torch
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

from window_presets import WINDOW_PRESETS
from ct_dataset_loader import is_ct_image_dicom
from windowing import apply_window


def _geometry_order_key(path: str):
	try:
		ds_hdr = pydicom.dcmread(path, stop_before_pixels=True, force=True)
		ipp = getattr(ds_hdr, 'ImagePositionPatient', None)
		iop = getattr(ds_hdr, 'ImageOrientationPatient', None)
		inst = getattr(ds_hdr, 'InstanceNumber', None)
		sloc = getattr(ds_hdr, 'SliceLocation', None)
		if iop is not None and len(iop) >= 6 and ipp is not None and len(ipp) >= 3:
			r = np.array([float(iop[0]), float(iop[1]), float(iop[2])], dtype=np.float64)
			c = np.array([float(iop[3]), float(iop[4]), float(iop[5])], dtype=np.float64)
			n = np.cross(r, c)
			p = np.array([float(ipp[0]), float(ipp[1]), float(ipp[2])], dtype=np.float64)
			zproj = float(np.dot(p, n))
			return (0, zproj, 0 if inst is None else int(inst))
		if ipp is not None and len(ipp) >= 3:
			return (1, float(ipp[2]), 0 if inst is None else int(inst))
		if sloc is not None:
			return (2, float(sloc), 0 if inst is None else int(inst))
		if inst is not None:
			return (3, float(int(inst)), 0)
	except Exception:
		pass
	return (4, 0.0, 0.0)


def _collect_ct_dicom_paths(folder_path: str):
	cand_paths = []
	for root, _, files in os.walk(folder_path):
		for f in files:
			if not f.lower().endswith('.dcm'):
				continue
			path = os.path.join(root, f)
			if is_ct_image_dicom(path):
				cand_paths.append(path)
	return cand_paths


def load_series_windowed(folder_path, preset="soft_tissue", override_window=None):
	"""Load CT series sorted by geometry, apply window/level, return [D,1,H,W] in [-1,1]."""
	window = WINDOW_PRESETS.get(preset, WINDOW_PRESETS["default"])
	if override_window is not None:
		wl, ww = override_window
	else:
		wl, ww = window["center"], window["width"]

	cand_paths = _collect_ct_dicom_paths(folder_path)
	if len(cand_paths) == 0:
		raise RuntimeError(f"No CT image DICOM files found under {folder_path}")

	slice_paths = sorted(cand_paths, key=_geometry_order_key)

	slice_list = []
	for path in slice_paths:
		try:
			ds = pydicom.dcmread(path, force=True)
			arr = ds.pixel_array
			hu = apply_modality_lut(arr, ds).astype(np.float32)
			if hu.ndim == 2:
				img = apply_window(hu, wl, ww)
				slice_list.append(torch.tensor(img).unsqueeze(0))
			elif hu.ndim == 3:
				for k in range(hu.shape[0]):
					img = apply_window(hu[k], wl, ww)
					slice_list.append(torch.tensor(img).unsqueeze(0))
		except Exception:
			continue

	if len(slice_list) == 0:
		raise RuntimeError(f"No readable CT image slices found under {folder_path}")

	H, W = slice_list[0].shape[-2:]
	slice_list = [s for s in slice_list if s.shape[-2:] == (H, W)]
	vol = torch.stack(slice_list, dim=0)
	return vol


def load_series_hu(folder_path):
	"""Load CT series in HU, sorted by geometry. Return ([D,1,H,W], meta_list)."""
	cand_paths = _collect_ct_dicom_paths(folder_path)
	if len(cand_paths) == 0:
		raise RuntimeError(f"No CT image DICOM files found under {folder_path}")

	slice_paths = sorted(cand_paths, key=_geometry_order_key)

	slice_list = []
	meta_list = []
	for path in slice_paths:
		try:
			ds = pydicom.dcmread(path, force=True)
			arr = ds.pixel_array
			hu = apply_modality_lut(arr, ds).astype(np.float32)
			if hu.ndim == 2:
				slice_list.append(torch.tensor(hu).unsqueeze(0))
				meta_list.append({
					'path': path,
					'InstanceNumber': getattr(ds, 'InstanceNumber', None),
					'SOPInstanceUID': str(getattr(ds, 'SOPInstanceUID', ''))
				})
			elif hu.ndim == 3:
				for k in range(hu.shape[0]):
					slice_list.append(torch.tensor(hu[k]).unsqueeze(0))
					meta_list.append({
						'path': path,
						'subindex': k,
						'InstanceNumber': getattr(ds, 'InstanceNumber', None),
						'SOPInstanceUID': str(getattr(ds, 'SOPInstanceUID', ''))
					})
		except Exception:
			continue

	if len(slice_list) == 0:
		raise RuntimeError(f"No CT image DICOM files found under {folder_path}")
	H, W = slice_list[0].shape[-2:]
	filtered = [(s, m) for s, m in zip(slice_list, meta_list) if s.shape[-2:] == (H, W)]
	if len(filtered) == 0:
		raise RuntimeError("No slices with consistent dimensions found")
	slice_list, meta_list = zip(*filtered)
	slice_list = list(slice_list)
	meta_list = list(meta_list)
	vol = torch.stack(slice_list, dim=0)
	return vol, meta_list


