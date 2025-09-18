import os, hashlib

def fixed_seed_for_path(path: str, base: int = 42) -> int:
    """
    Path independent, but patient specific seed:
    - uses only the folder name (e.g. '12196pp')
    - normalized (lowercase) for stability on Windows/macOS
    - combined with base seed, so that Val/Test get different seeds
    """
    patient_dirname = os.path.basename(os.path.normpath(path)).lower()
    key = f"{int(base)}|{patient_dirname}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # 32-bit Seed


