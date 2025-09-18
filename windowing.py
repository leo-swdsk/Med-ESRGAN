import numpy as np


def apply_window(img, center, width):
	min_val = center - width / 2.0
	max_val = center + width / 2.0
	img = np.clip(img.astype(np.float32), min_val, max_val)
	img = (img - min_val) / (max_val - min_val) # [0,1]
	img = img * 2.0 - 1.0 # [-1,1]
	return img.astype(np.float32)


