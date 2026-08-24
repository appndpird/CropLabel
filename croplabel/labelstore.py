"""Label persistence for CropLabel.

Per input image (same RELATIVE PATH and FILENAME as the input, mirrored under
the output root):
  <name>.png             semantic index mask at the ORIGINAL image size
                         (0=unlabeled, 1=soil, 2=crop, 3=weed)
  <name>_instances.png   uint16 instance-ID map (0 = no instance) — instance
                         segmentation companion to the semantic mask
  <name>_color.png       color-coded semantic mask
  <name>_overlay.png     semantic mask blended over the image
  <name>.json            metadata: crop type, status, per-instance records
                         (id, class, score, area, bbox)
records.json             queue status store at the output root
"""
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .config import (CLS_CROP, CLS_SOIL, CLS_WEED, CLASS_COLORS_BGR)


def colorize(sem: np.ndarray) -> np.ndarray:
    out = np.zeros((*sem.shape, 3), np.uint8)
    for cid, bgr in CLASS_COLORS_BGR.items():
        out[sem == cid] = bgr
    return out


class LabelStore:
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.records_path = self.root / "records.json"
        self.lock = threading.Lock()

    # ------------------------------------------------------------- records
    def _read_records(self) -> dict:
        if self.records_path.exists():
            try:
                return json.loads(self.records_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def records(self) -> dict:
        with self.lock:
            return self._read_records()

    def _update_record(self, key: str, rec: dict):
        with self.lock:
            all_r = self._read_records()
            all_r[key] = rec
            self.records_path.write_text(json.dumps(all_r, indent=1),
                                         encoding="utf-8")

    def status_of(self, key: str) -> str:
        return self.records().get(key, {}).get("status", "pending")

    def mark_skipped(self, key: str):
        self._update_record(key, {"status": "skipped",
                                  "labeled_at": time.strftime("%Y-%m-%d %H:%M:%S")})

    # --------------------------------------------------------------- paths
    def paths_for(self, key: str) -> dict:
        rel = Path(key)
        d = self.root / rel.parent
        stem = rel.stem
        return {
            "dir": d,
            "semantic": d / rel.name,                      # SAME filename
            "instances": d / f"{stem}_instances.png",
            "color": d / f"{stem}_color.png",
            "overlay": d / f"{stem}_overlay.png",
            "meta": d / f"{stem}.json",
        }

    # ---------------------------------------------------------------- load
    def load(self, key: str, work_hw=None):
        """(semantic, instance_map, meta) at work resolution, or (None,)*3."""
        p = self.paths_for(key)
        if not p["semantic"].exists():
            return None, None, None
        sem = cv2.imread(str(p["semantic"]), cv2.IMREAD_GRAYSCALE)
        inst = cv2.imread(str(p["instances"]), cv2.IMREAD_UNCHANGED)
        meta = {}
        if p["meta"].exists():
            try:
                meta = json.loads(p["meta"].read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        if work_hw and sem is not None and sem.shape != tuple(work_hw):
            sem = cv2.resize(sem, (work_hw[1], work_hw[0]),
                             interpolation=cv2.INTER_NEAREST)
            if inst is not None:
                inst = cv2.resize(inst, (work_hw[1], work_hw[0]),
                                  interpolation=cv2.INTER_NEAREST)
        return sem, inst, meta

    # ---------------------------------------------------------------- save
    def save(self, key: str, img_path: str, sem_work: np.ndarray,
             inst_work: np.ndarray, instances_meta: list, crop_type: str,
             engine: str, status: str = "reviewed") -> dict:
        """Upscale to the original image size and write every output file."""
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cannot read {img_path}")
        H, W = img.shape[:2]
        sem = cv2.resize(sem_work.astype(np.uint8), (W, H),
                         interpolation=cv2.INTER_NEAREST)
        inst = cv2.resize(inst_work.astype(np.uint16), (W, H),
                          interpolation=cv2.INTER_NEAREST)

        p = self.paths_for(key)
        p["dir"].mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p["semantic"]), sem)
        cv2.imwrite(str(p["instances"]), inst)
        color = colorize(sem)
        cv2.imwrite(str(p["color"]), color)
        overlay = cv2.addWeighted(img, 0.65, color, 0.35, 0)
        cv2.imwrite(str(p["overlay"]), overlay)

        # scale instance meta bboxes/areas to original resolution
        sx, sy = W / sem_work.shape[1], H / sem_work.shape[0]
        inst_out = []
        for m in instances_meta:
            b = m.get("bbox")
            inst_out.append({**m,
                "bbox": [b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy] if b else None,
                "area_px": int(np.count_nonzero(inst == m["id"]))})

        n_crop = sum(1 for m in inst_out if m["class"] == CLS_CROP)
        n_weed = sum(1 for m in inst_out if m["class"] == CLS_WEED)
        meta = {
            "key": key, "image": img_path, "width": W, "height": H,
            "crop_type": crop_type, "status": status, "engine": engine,
            "labeled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_crop_instances": n_crop, "n_weed_instances": n_weed,
            "crop_pct": round(100.0 * float((sem == CLS_CROP).mean()), 3),
            "weed_pct": round(100.0 * float((sem == CLS_WEED).mean()), 3),
            "soil_pct": round(100.0 * float((sem == CLS_SOIL).mean()), 3),
            "instances": inst_out,
        }
        p["meta"].write_text(json.dumps(meta, indent=1), encoding="utf-8")

        rec = {"status": status, "engine": engine, "crop_type": crop_type,
               "n_crop_instances": n_crop, "n_weed_instances": n_weed,
               "crop_pct": meta["crop_pct"], "weed_pct": meta["weed_pct"],
               "labeled_at": meta["labeled_at"], "out_dir": str(p["dir"])}
        self._update_record(key, rec)
        return rec
