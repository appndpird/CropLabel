"""Vegetation / soil separation and dataset scanning.

Soil is labeled automatically: everything below the ExG (Excess Green index)
threshold is soil (including stubble/straw, which is dead material). The
vegetation mask also gates SAM instances so masks never bleed into soil.
"""
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def scan_dataset(root: Path) -> dict:
    """{key: absolute path} for every image under root (recursive).
    key = relative path with '/' separators, extension kept so the label
    file can mirror the exact input filename."""
    out = {}
    root = Path(root)
    if not root.is_dir():
        return out
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            out[f.relative_to(root).as_posix()] = str(f)
    return out


def exg_index(bgr: np.ndarray) -> np.ndarray:
    """Excess Green index in [-1, 1] (normalized rgb: 2g - r - b)."""
    b, g, r = [c.astype(np.float32) for c in cv2.split(bgr)]
    s = b + g + r + 1e-6
    return 2 * g / s - r / s - b / s


def vegetation_mask(bgr: np.ndarray, thresh: float = 0.06,
                    min_px: int = 8) -> np.ndarray:
    """Bool mask of living (green) vegetation. Stubble and soil are False."""
    m = (exg_index(bgr) > thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if min_px > 1:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = np.zeros_like(m, bool)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_px:
                keep |= lab == i
        return keep
    return m > 0


def soil_mask(bgr: np.ndarray, thresh: float = 0.06) -> np.ndarray:
    """Soil = everything that is not living vegetation (stubble counts as
    soil background for cropping purposes)."""
    veg = vegetation_mask(bgr, thresh)
    # close small gaps inside plants so soil doesn't leak into leaf interiors
    veg_d = cv2.dilate(veg.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return ~veg_d
