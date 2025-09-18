import pydicom
import numpy as np
import torch
from pydicom.pixel_data_handlers.util import apply_modality_lut

def load_ct_as_tensor(dicom_path, hu_clip=(-1000, 2000)):
    # load DICOM and convert to HU (Modality LUT)
    ds = pydicom.dcmread(dicom_path, force=True)
    arr = ds.pixel_array
    try:
        hu = apply_modality_lut(arr, ds).astype(np.float32)
    except Exception:
        hu = arr.astype(np.float32)

    lo, hi = hu_clip
    img = np.clip(hu, lo, hi)
    img = (img - lo) / (hi - lo)
    img = img * 2 - 1

    tensor = torch.tensor(img).unsqueeze(0)  # [1, H, W]
    return tensor
