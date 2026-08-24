"""Adaptive crop-vs-weed instance classifier.

SAM3 finds every PLANT, but crop vs weed is a semantic distinction the text
prompt cannot make reliably. This classifier learns it from the user's
reviewed labels: every saved instance (with its class) becomes a training
sample. Features are simple, fast and work at any resolution: color
statistics (HSV + ExG), shape (area, aspect, solidity, elongation) and
texture. cv2.ml RTrees keeps it dependency-free and retrains in <1 s.

While fewer than MIN_SAMPLES weed examples exist, everything defaults to
crop (fields contain one crop; weeds are the exception) and the classifier
abstains.
"""
import json
import threading
from pathlib import Path

import cv2
import numpy as np

from .config import APP_DIR, CLS_CROP, CLS_WEED
from .vegetation import exg_index

MODEL_PATH = APP_DIR / "models" / "crop_weed_rtrees.yml"
MIN_SAMPLES = 3          # need at least this many of EACH class to train


def instance_features(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """13-dim feature vector for one instance mask on the image."""
    m = mask.astype(bool)
    if not m.any():
        return np.zeros(13, np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    exg = exg_index(bgr)
    px_hsv = hsv[m].astype(np.float32)
    px_exg = exg[m]
    ys, xs = np.nonzero(m)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    area = float(m.sum())
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    hull_area = per = 0.0
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        hull_area = float(cv2.contourArea(cv2.convexHull(c))) or 1.0
        per = float(cv2.arcLength(c, True)) or 1.0
    lap = cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_32F)
    return np.array([
        px_hsv[:, 0].mean(), px_hsv[:, 0].std(),      # hue
        px_hsv[:, 1].mean(), px_hsv[:, 1].std(),      # saturation
        px_hsv[:, 2].mean(),                          # value
        px_exg.mean(), px_exg.std(),                  # greenness
        np.log1p(area),
        w / max(h, 1),                                # aspect
        area / max(hull_area, 1.0),                   # solidity
        per / np.sqrt(max(area, 1.0)),                # boundary complexity
        float(lap[m].std()),                          # texture
        area / max(h * w, 1),                         # bbox fill
    ], np.float32)


class CropWeedClassifier:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.trained_on = 0
        self._last_counts = (0, 0)
        if MODEL_PATH.exists():
            try:
                self.model = cv2.ml.RTrees_load(str(MODEL_PATH))
            except Exception:
                self.model = None

    def status(self) -> dict:
        return {"active": self.model is not None,
                "trained_on": self.trained_on,
                "crop_samples": self._last_counts[0],
                "weed_samples": self._last_counts[1]}

    # ------------------------------------------------------------ training
    def collect_training(self, store, images: dict, work_res: int):
        """(X, y) from every reviewed image's instances."""
        X, y = [], []
        recs = store.records()
        for key, r in recs.items():
            if r.get("status") != "reviewed" or key not in images:
                continue
            p = store.paths_for(key)
            if not (p["instances"].exists() and p["meta"].exists()):
                continue
            try:
                meta = json.loads(p["meta"].read_text(encoding="utf-8"))
            except Exception:
                continue
            img = cv2.imread(images[key], cv2.IMREAD_COLOR)
            inst = cv2.imread(str(p["instances"]), cv2.IMREAD_UNCHANGED)
            if img is None or inst is None:
                continue
            s = work_res / max(img.shape[:2])
            small = cv2.resize(img, None, fx=s, fy=s,
                               interpolation=cv2.INTER_AREA)
            inst_s = cv2.resize(inst, (small.shape[1], small.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            for m in meta.get("instances", []):
                if m["class"] not in (CLS_CROP, CLS_WEED):
                    continue
                mask = inst_s == m["id"]
                if mask.sum() < 10:
                    continue
                X.append(instance_features(small, mask))
                y.append(m["class"])
        return (np.array(X, np.float32) if X else np.zeros((0, 13), np.float32),
                np.array(y, np.int32))

    def maybe_retrain(self, store, images: dict, work_res: int) -> bool:
        X, y = self.collect_training(store, images, work_res)
        n_crop = int((y == CLS_CROP).sum())
        n_weed = int((y == CLS_WEED).sum())
        counts = (n_crop, n_weed)
        with self.lock:
            if counts == self._last_counts:
                return False
            self._last_counts = counts
            if n_crop < MIN_SAMPLES or n_weed < MIN_SAMPLES:
                return False
            rt = cv2.ml.RTrees_create()
            rt.setMaxDepth(12)
            rt.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 120, 1e-6))
            rt.train(X, cv2.ml.ROW_SAMPLE, y)
            self.model = rt
            self.trained_on = len(y)
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            rt.save(str(MODEL_PATH))
            return True

    # ----------------------------------------------------------- inference
    def classify(self, bgr: np.ndarray, mask: np.ndarray) -> int:
        """CLS_CROP or CLS_WEED (defaults to crop when untrained)."""
        if self.model is None:
            return CLS_CROP
        f = instance_features(bgr, mask).reshape(1, -1)
        _, out = self.model.predict(f)
        return int(out[0][0])
