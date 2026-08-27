"""Label persistence for CropLabel.

Per input image (same RELATIVE PATH and FILENAME as the input, mirrored under
the output root):
  <name>.png             semantic index mask at the ORIGINAL image size
                         (0=unlabeled, 1=soil, 2=crop, 3=weed)
  <name>_instances.png   uint16 instance-ID map (0 = no instance) — instance
                         segmentation companion to the semantic mask
  <name>_color.png       color-coded semantic mask
  <name>_overlay.png     semantic mask blended over the image (+ ids / boxes
                         when those display options are on)
  <name>.json            metadata: crop type, status, per-instance records
                         (id, class, score, area, bbox, centroid) + summary
                         statistics (counts, plants/m^2, greenness ratio ...)
  <name>_boxes.json      [optional, save_boxes] per-instance bounding boxes
  <name>_boxes.txt       [optional] YOLO format: cls cx cy w h (normalized)
                         class 0 = crop, 1 = weed
  <name>_boxes.png       [optional] image with boxes drawn
records.json             queue status store at the output root
summary.csv              one row per labeled image (all summary statistics)
"""
import csv
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .config import (CLS_CROP, CLS_SOIL, CLS_UNLABELED, CLS_WEED,
                     CLASS_COLORS_BGR)
from .vegetation import exg_index


def colorize(sem: np.ndarray) -> np.ndarray:
    out = np.zeros((*sem.shape, 3), np.uint8)
    for cid, bgr in CLASS_COLORS_BGR.items():
        out[sem == cid] = bgr
    return out


def draw_annotations(img: np.ndarray, instances: list, show_ids: bool,
                     show_boxes: bool, selected: int | None = None,
                     scale: float = 1.0) -> np.ndarray:
    """Draw bounding boxes and/or instance numbers on `img` (in place).
    `instances` bboxes are in the coordinate system of `img`."""
    h, w = img.shape[:2]
    fs = max(0.35, 0.45 * scale)
    th = max(1, int(round(scale)))
    for m in instances:
        b = m.get("bbox")
        if not b:
            continue
        x0, y0, x1, y1 = (int(round(v)) for v in b)
        col = CLASS_COLORS_BGR.get(int(m["class"]), (255, 255, 255))
        sel = m.get("id") == selected
        if sel:
            col = (0, 255, 255)
        if show_boxes:
            cv2.rectangle(img, (x0, y0), (x1, y1), col, th + (1 if sel else 0))
        if show_ids:
            txt = str(m["id"])
            (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            tx = max(0, min(w - tw - 2, x0))
            ty = y0 - 3 if y0 - tht - 4 >= 0 else min(h - 2, y1 + tht + 3)
            cv2.rectangle(img, (tx - 1, ty - tht - 2), (tx + tw + 1, ty + 2),
                          (0, 0, 0), -1)
            cv2.putText(img, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        col, th, cv2.LINE_AA)
    return img


def image_stats(img: np.ndarray, sem: np.ndarray, inst_out: list,
                exg_thresh: float, gsd_mm: float | None) -> dict:
    """Summary statistics for one labeled image (original resolution)."""
    H, W = sem.shape[:2]
    exg = exg_index(img)
    veg = exg > exg_thresh
    n_crop = sum(1 for m in inst_out if m["class"] == CLS_CROP)
    n_weed = sum(1 for m in inst_out if m["class"] == CLS_WEED)
    crop_px = int((sem == CLS_CROP).sum())
    weed_px = int((sem == CLS_WEED).sum())
    crop_areas = [m["area_px"] for m in inst_out if m["class"] == CLS_CROP]
    labeled = (sem == CLS_CROP) | (sem == CLS_WEED)
    st = {
        "n_instances": len(inst_out),
        "n_crop_instances": n_crop, "n_weed_instances": n_weed,
        "crop_pct": round(100.0 * crop_px / (H * W), 3),
        "weed_pct": round(100.0 * weed_px / (H * W), 3),
        "soil_pct": round(100.0 * float((sem == CLS_SOIL).mean()), 3),
        "unlabeled_pct": round(100.0 * float((sem == CLS_UNLABELED).mean()), 3),
        # greenness: fraction of pixels that are living vegetation (ExG) and
        # the mean ExG index over the image / over the crop pixels
        "greenness_ratio": round(float(veg.mean()), 4),
        "mean_exg": round(float(exg.mean()), 4),
        "mean_exg_crop": round(float(exg[sem == CLS_CROP].mean()), 4) if crop_px else None,
        "labeled_veg_fraction": round(float(labeled[veg].mean()), 4) if veg.any() else None,
        "mean_crop_area_px": round(float(np.mean(crop_areas)), 1) if crop_areas else None,
        "median_crop_area_px": round(float(np.median(crop_areas)), 1) if crop_areas else None,
        "gsd_mm_per_px": gsd_mm,
        "image_area_m2": None, "plants_per_m2": None, "weeds_per_m2": None,
        "mean_crop_area_cm2": None, "crop_cover_m2": None,
    }
    if gsd_mm:
        px_m2 = (float(gsd_mm) / 1000.0) ** 2
        area = H * W * px_m2
        st.update({
            "image_area_m2": round(area, 4),
            "plants_per_m2": round(n_crop / area, 2) if area else None,
            "weeds_per_m2": round(n_weed / area, 2) if area else None,
            "crop_cover_m2": round(crop_px * px_m2, 4),
            "mean_crop_area_cm2": round(float(np.mean(crop_areas)) * px_m2 * 1e4, 2) if crop_areas else None,
        })
    return st


SUMMARY_COLS = ["key", "status", "crop_type", "engine", "labeled_at", "width",
                "height", "n_instances", "n_crop_instances", "n_weed_instances",
                "plants_per_m2", "weeds_per_m2", "image_area_m2",
                "crop_pct", "weed_pct", "soil_pct", "unlabeled_pct",
                "greenness_ratio", "mean_exg", "mean_exg_crop",
                "labeled_veg_fraction", "mean_crop_area_px",
                "median_crop_area_px", "mean_crop_area_cm2", "crop_cover_m2",
                "gsd_mm_per_px"]


class LabelStore:
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.records_path = self.root / "records.json"
        self.summary_path = self.root / "summary.csv"
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
            self._write_summary(all_r)

    def _write_summary(self, all_r: dict):
        rows = [{"key": k, **v} for k, v in sorted(all_r.items())
                if v.get("status") in ("reviewed", "auto")]
        try:
            with open(self.summary_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=SUMMARY_COLS, extrasaction="ignore")
                w.writeheader()
                for r in rows:
                    w.writerow(r)
        except OSError:
            pass   # e.g. the CSV is open in Excel — records.json is authoritative

    def summary(self) -> dict:
        """Aggregate over all labeled images (for the UI summary panel)."""
        recs = [v for v in self.records().values()
                if v.get("status") in ("reviewed", "auto")]

        def _mean(k):
            vals = [r[k] for r in recs if r.get(k) is not None]
            return round(float(np.mean(vals)), 3) if vals else None
        return {
            "n_images": len(recs),
            "n_reviewed": sum(1 for r in recs if r["status"] == "reviewed"),
            "n_auto": sum(1 for r in recs if r["status"] == "auto"),
            "total_crop": sum(r.get("n_crop_instances", 0) for r in recs),
            "total_weed": sum(r.get("n_weed_instances", 0) for r in recs),
            "mean_crop_per_image": _mean("n_crop_instances"),
            "mean_plants_per_m2": _mean("plants_per_m2"),
            "mean_weeds_per_m2": _mean("weeds_per_m2"),
            "mean_greenness_ratio": _mean("greenness_ratio"),
            "mean_crop_pct": _mean("crop_pct"),
            "mean_weed_pct": _mean("weed_pct"),
            "summary_csv": str(self.summary_path),
        }

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
            "boxes_json": d / f"{stem}_boxes.json",
            "boxes_txt": d / f"{stem}_boxes.txt",
            "boxes_png": d / f"{stem}_boxes.png",
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
             engine: str, status: str = "reviewed",
             options: dict | None = None) -> dict:
        """Upscale to the original image size and write every output file.

        options: show_ids, show_boxes, save_boxes, exg_thresh, gsd_mm_per_px
        """
        o = options or {}
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

        # per-instance records at ORIGINAL resolution (bbox recomputed from
        # the upscaled instance map so it is exact)
        inst_out = []
        for m in instances_meta:
            ys, xs = np.nonzero(inst == m["id"])
            if len(xs) == 0:
                continue
            inst_out.append({**m,
                "bbox": [int(xs.min()), int(ys.min()),
                         int(xs.max() + 1), int(ys.max() + 1)],
                "centroid": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
                "area_px": int(len(xs))})

        scale = max(1.0, max(H, W) / 1024.0)
        overlay = cv2.addWeighted(img, 0.65, color, 0.35, 0)
        draw_annotations(overlay, inst_out, bool(o.get("show_ids")),
                         bool(o.get("show_boxes")), scale=scale)
        cv2.imwrite(str(p["overlay"]), overlay)

        if o.get("save_boxes"):
            self._write_boxes(p, img, inst_out, W, H, scale)
        else:
            for k in ("boxes_json", "boxes_txt", "boxes_png"):
                if p[k].exists():
                    p[k].unlink()

        stats = image_stats(img, sem, inst_out,
                            float(o.get("exg_thresh", 0.06)),
                            o.get("gsd_mm_per_px"))
        labeled_at = time.strftime("%Y-%m-%d %H:%M:%S")
        meta = {
            "key": key, "image": img_path, "width": W, "height": H,
            "crop_type": crop_type, "status": status, "engine": engine,
            "labeled_at": labeled_at,
            **stats,
            "boxes_file": str(p["boxes_json"]) if o.get("save_boxes") else None,
            "instances": inst_out,
        }
        p["meta"].write_text(json.dumps(meta, indent=1), encoding="utf-8")

        rec = {"status": status, "engine": engine, "crop_type": crop_type,
               "labeled_at": labeled_at, "width": W, "height": H,
               **stats, "out_dir": str(p["dir"])}
        self._update_record(key, rec)
        return rec

    @staticmethod
    def _write_boxes(p: dict, img: np.ndarray, inst_out: list, W: int, H: int,
                     scale: float):
        boxes = [{"id": m["id"], "class": m["class"],
                  "label": "crop" if m["class"] == CLS_CROP else "weed",
                  "bbox_xyxy": m["bbox"], "area_px": m["area_px"],
                  "score": m.get("score")} for m in inst_out]
        p["boxes_json"].write_text(json.dumps(
            {"width": W, "height": H, "format": "xyxy pixels",
             "classes": {0: "crop", 1: "weed"}, "boxes": boxes}, indent=1),
            encoding="utf-8")
        lines = []
        for m in inst_out:
            x0, y0, x1, y1 = m["bbox"]
            c = 0 if m["class"] == CLS_CROP else 1
            lines.append(f"{c} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                         f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
        p["boxes_txt"].write_text("\n".join(lines) + ("\n" if lines else ""),
                                  encoding="utf-8")
        vis = img.copy()
        draw_annotations(vis, inst_out, True, True, scale=scale)
        cv2.imwrite(str(p["boxes_png"]), vis)
