"""CropLabel FastAPI server — agentic crop/weed/soil labeling with SAM3.1.

Design: the server owns the editing state (semantic mask + instance map per
open image). The browser sends ACTIONS (exemplar click, delete, brush, split,
polygon, ...) and receives a freshly rendered overlay. All segmentation,
gating and learning happens here in Python.

Review workflow: ✨ auto-label (SAM3) -> review -> fix with the manual tools
(select / polygon / brush-into-instance / split by line, seeds or box /
merge / delete / reclassify) -> Save (writes masks, json, optional bbox files
and the summary statistics).
"""
import base64
import json
import threading
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import config, editing
from .config import CLS_CROP, CLS_SOIL, CLS_UNLABELED, CLS_WEED
from .classifier import CropWeedClassifier
from .labelstore import LabelStore, colorize, draw_annotations
from .sam_engine import Sam3NativeEngine
from .vegetation import scan_dataset, vegetation_mask

app = FastAPI(title="CropLabel")
STATIC = Path(__file__).parent / "static"

settings = config.load_settings()
engine = Sam3NativeEngine(settings["sam3_ckpt"],
                          float(settings.get("native_threshold", 0.4)))
clf = CropWeedClassifier()
_store = None
_store_lock = threading.Lock()

VIEW_KEYS = ("show_ids", "show_boxes", "save_boxes", "side_by_side",
             "snap_polygon_to_veg")


def store() -> LabelStore:
    global _store
    with _store_lock:
        if _store is None or str(_store.root) != settings["output_root"]:
            _store = LabelStore(settings["output_root"])
        return _store


def images() -> dict:
    return scan_dataset(Path(settings["dataset_root"]))


def png_b64(arr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", arr)
    return base64.b64encode(buf.tobytes()).decode()


def jpg_b64(arr: np.ndarray, q: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode()


def save_options() -> dict:
    return {"show_ids": settings.get("show_ids", False),
            "show_boxes": settings.get("show_boxes", False),
            "save_boxes": settings.get("save_boxes", False),
            "exg_thresh": float(settings["exg_thresh"]),
            "gsd_mm_per_px": settings.get("gsd_mm_per_px")}


# ---------------------------------------------------------------------------
# Editing state (server-side, one image at a time)
# ---------------------------------------------------------------------------
class EditState:
    def __init__(self, key: str, img_path: str):
        self.key = key
        self.img_path = img_path
        big = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if big is None:
            raise RuntimeError(f"cannot read {img_path}")
        res = int(settings.get("work_res", 1024))
        s = res / max(big.shape[:2])
        self.img = cv2.resize(big, (max(2, round(big.shape[1] * s)),
                                    max(2, round(big.shape[0] * s))),
                              interpolation=cv2.INTER_AREA if s < 1
                              else cv2.INTER_LANCZOS4)
        self.h, self.w = self.img.shape[:2]
        self.veg = vegetation_mask(self.img, float(settings["exg_thresh"]))
        self.sem = np.where(self.veg, CLS_UNLABELED, CLS_SOIL).astype(np.uint8)
        self.inst = np.zeros((self.h, self.w), np.uint16)
        self.meta = {}          # id -> {class, score, source, bbox}
        self.next_id = 1
        self.crop_type = settings.get("default_crop_type", "barley")
        self.pos_boxes, self.neg_boxes = [], []   # exemplar memory (this img)
        self.undo, self.redo = [], []
        self.dirty = False
        self.selected: int | None = None          # instance being edited

    # ------------------------------------------------------------- history
    def push_undo(self):
        self.undo.append((self.sem.copy(), self.inst.copy(),
                          {k: dict(v) for k, v in self.meta.items()},
                          self.next_id))
        if len(self.undo) > 25:
            self.undo.pop(0)
        self.redo = []

    def do_undo(self):
        if not self.undo:
            return False
        self.redo.append((self.sem, self.inst, self.meta, self.next_id))
        self.sem, self.inst, self.meta, self.next_id = self.undo.pop()
        self._check_selected()
        return True

    def do_redo(self):
        if not self.redo:
            return False
        self.undo.append((self.sem, self.inst, self.meta, self.next_id))
        self.sem, self.inst, self.meta, self.next_id = self.redo.pop()
        self._check_selected()
        return True

    def _check_selected(self):
        if self.selected is not None and self.selected not in self.meta:
            self.selected = None

    # ----------------------------------------------------------- instances
    def refresh_meta(self, iid: int):
        """Recompute bbox/area after the instance mask changed."""
        st = editing.instance_stats(self.inst == iid)
        if st["area_px"] == 0:
            self.meta.pop(iid, None)
            if self.selected == iid:
                self.selected = None
            return
        self.meta[iid]["bbox"] = st["bbox"]
        self.meta[iid]["area_px"] = st["area_px"]

    def add_instance(self, mask: np.ndarray, cls: int, score: float,
                     source: str, gate: bool | None = None) -> int | None:
        gate = settings.get("veg_gate", True) if gate is None else gate
        if gate:
            g = cv2.dilate(self.veg.astype(np.uint8),
                           np.ones((5, 5), np.uint8)) > 0
            mask = mask & g
        if mask.sum() < int(settings.get("min_instance_px", 30)):
            return None
        # claim only pixels not already owned by another instance
        mask = mask & (self.inst == 0)
        if mask.sum() < int(settings.get("min_instance_px", 30)):
            return None
        iid = self.next_id
        self.next_id += 1
        self.inst[mask] = iid
        self.sem[mask] = cls
        self.meta[iid] = {"id": iid, "class": int(cls),
                          "score": round(float(score), 3), "source": source}
        self.refresh_meta(iid)
        return iid

    def add_part(self, mask: np.ndarray, cls: int, source: str) -> int:
        """Register an already-owned mask as a new instance (used by split;
        no gating, no minimum)."""
        iid = self.next_id
        self.next_id += 1
        self.inst[mask] = iid
        self.sem[mask] = cls
        self.meta[iid] = {"id": iid, "class": int(cls), "score": 1.0,
                          "source": source}
        self.refresh_meta(iid)
        return iid

    def remove_instance(self, iid: int):
        m = self.inst == iid
        self.inst[m] = 0
        self.sem[m] = np.where(self.veg[m], CLS_UNLABELED, CLS_SOIL)
        self.meta.pop(iid, None)
        if self.selected == iid:
            self.selected = None

    def reclass_instance(self, iid: int, cls: int):
        m = self.inst == iid
        self.sem[m] = cls
        if iid in self.meta:
            self.meta[iid]["class"] = int(cls)

    def grow_instance(self, iid: int, mask: np.ndarray, steal: bool = False):
        """Add pixels to an existing instance (brush / polygon into selected).
        By default pixels owned by OTHER instances are left alone."""
        if iid not in self.meta:
            return 0
        if not steal:
            mask = mask & ((self.inst == 0) | (self.inst == iid))
        n = int((mask & (self.inst != iid)).sum())
        self.inst[mask] = iid
        self.sem[mask] = self.meta[iid]["class"]
        self.refresh_meta(iid)
        return n

    def shrink_instance(self, iid: int, mask: np.ndarray):
        m = mask & (self.inst == iid)
        self.inst[m] = 0
        self.sem[m] = np.where(self.veg[m], CLS_UNLABELED, CLS_SOIL)
        self.refresh_meta(iid)
        return int(m.sum())

    def split_instance(self, iid: int, parts: list, source: str) -> list:
        """Replace instance `iid` by `parts` (list of bool masks). The largest
        part keeps the original id."""
        if iid not in self.meta or len(parts) < 2:
            return [iid]
        cls = self.meta[iid]["class"]
        parts = sorted(parts, key=lambda p: -int(p.sum()))
        whole = self.inst == iid
        self.inst[whole] = 0
        self.inst[parts[0]] = iid
        self.meta[iid]["source"] = source
        self.refresh_meta(iid)
        ids = [iid]
        for p in parts[1:]:
            ids.append(self.add_part(p, cls, source))
        # any leftover pixels of the old instance not covered by a part
        leftover = whole & (self.inst == 0)
        if leftover.any():
            self.sem[leftover] = np.where(self.veg[leftover], CLS_UNLABELED, CLS_SOIL)
        # number the parts iid, iid+1, ... and shift everything after them
        rest = [i for i in sorted(self.meta) if i not in ids]
        pos = sum(1 for i in rest if i < iid)
        order = rest[:pos] + ids + rest[pos:]
        self.renumber(order)
        return list(range(order.index(iid) + 1, order.index(iid) + 1 + len(ids)))

    def renumber(self, order: list) -> None:
        """Relabel the instances to 1..N in the given order (current ids).
        Keeps ids sequential after splits / merges / deletes so the numbers
        drawn on the image and written to the files never have gaps."""
        mapping = {old: i + 1 for i, old in enumerate(order)}
        if len(mapping) != len(self.meta) or all(k == v for k, v in mapping.items()):
            if len(mapping) == len(self.meta):
                self.next_id = len(order) + 1
                return
        lut = np.zeros(max(int(self.next_id), int(self.inst.max()) + 1), np.uint16)
        for k, v in mapping.items():
            lut[k] = v
        self.inst = lut[self.inst]
        self.meta = {mapping[k]: dict(v, id=mapping[k]) for k, v in self.meta.items()}
        self.selected = mapping.get(self.selected) if self.selected is not None else None
        self.next_id = len(order) + 1

    def compact(self) -> None:
        """Close the gaps left by delete / merge / erase (keeps the order)."""
        self.renumber(sorted(self.meta))

    def merge_instances(self, keep: int, other: int):
        if keep not in self.meta or other not in self.meta or keep == other:
            return False
        m = self.inst == other
        self.inst[m] = keep
        self.sem[m] = self.meta[keep]["class"]
        self.meta.pop(other, None)
        self.refresh_meta(keep)
        return True

    def instance_at(self, x, y) -> int:
        if x is None or y is None:
            return 0
        xi, yi = int(x), int(y)
        if 0 <= yi < self.h and 0 <= xi < self.w:
            return int(self.inst[yi, xi])
        return 0

    def instances_meta(self) -> list:
        return [dict(v) for v in self.meta.values()]

    def counts(self) -> dict:
        return {"n_instances": len(self.meta),
                "n_crop": sum(1 for m in self.meta.values() if m["class"] == CLS_CROP),
                "n_weed": sum(1 for m in self.meta.values() if m["class"] == CLS_WEED)}

    def selected_info(self) -> dict | None:
        if self.selected is None or self.selected not in self.meta:
            return None
        m = self.meta[self.selected]
        return {"id": self.selected, "class": m["class"],
                "class_name": "crop" if m["class"] == CLS_CROP else "weed",
                "area_px": m.get("area_px"), "bbox": m.get("bbox"),
                "source": m.get("source")}

    # ------------------------------------------------------------ rendering
    def overlay(self) -> np.ndarray:
        color = colorize(self.sem)
        out = cv2.addWeighted(self.img, 1.0, color, 0.45, 0)
        # instance boundaries so single plants are visibly separated
        ii = self.inst
        edges = np.zeros((self.h, self.w), bool)
        edges[:-1, :] |= (ii[:-1, :] != ii[1:, :])
        edges[:, :-1] |= (ii[:, :-1] != ii[:, 1:])
        edges &= ii > 0
        out[edges] = (255, 255, 255)
        # selected instance: thick yellow outline
        if self.selected is not None and self.selected in self.meta:
            sel = (ii == self.selected).astype(np.uint8)
            cnts, _ = cv2.findContours(sel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, (0, 255, 255), 2)
        draw_annotations(out, list(self.meta.values()),
                         bool(settings.get("show_ids")),
                         bool(settings.get("show_boxes")),
                         selected=self.selected,
                         scale=max(1.0, max(self.h, self.w) / 1024.0))
        return out


_states: dict[str, EditState] = {}
_state_lock = threading.Lock()


def state_of(key: str, fresh: bool = False) -> EditState:
    with _state_lock:
        if not fresh and key in _states:
            return _states[key]
        imgs = images()
        if key not in imgs:
            raise KeyError(key)
        st = EditState(key, imgs[key])
        # load saved labels if any
        sem, inst, meta = store().load(key, work_hw=(st.h, st.w))
        if sem is not None:
            st.sem = sem.astype(np.uint8)
            st.inst = (inst if inst is not None
                       else np.zeros((st.h, st.w), np.uint16)).astype(np.uint16)
            for m in (meta or {}).get("instances", []):
                iid = int(m["id"])
                st.meta[iid] = {"id": iid, "class": int(m["class"]),
                                "score": m.get("score", 1.0),
                                "source": m.get("source", "saved")}
                st.refresh_meta(iid)
            # instances present in the map but missing from the json
            for iid in np.unique(st.inst):
                iid = int(iid)
                if iid and iid not in st.meta:
                    cls = int(np.bincount(st.sem[st.inst == iid]).argmax())
                    st.meta[iid] = {"id": iid, "class": cls if cls in (CLS_CROP, CLS_WEED) else CLS_CROP,
                                    "score": 1.0, "source": "saved"}
                    st.refresh_meta(iid)
            st.next_id = max(list(st.meta) + [0]) + 1
            st.crop_type = (meta or {}).get("crop_type", st.crop_type)
        if len(_states) > 2:
            for k in [k for k in _states if k != key][:-1]:
                _states.pop(k, None)
        _states[key] = st
        return st


# ---------------------------------------------------------------------------
# SAM helpers
# ---------------------------------------------------------------------------
def _dedupe(instances: list, cover: float = 0.6, min_px: int = 30) -> list:
    """Resolve overlapping SAM detections, preferring the FINER partition.

    SAM3 often returns both a cluster mask (2-4 neighbouring seedlings) and
    the individual plants inside it. Scanning smallest-first:
      * a detection covered (> `cover`) by >= 2 already-kept smaller ones is
        a cluster of plants we already have -> only its uncovered remainder
        is kept (a third plant SAM did not detect on its own);
      * kept detections lying mostly inside the current one when it is NOT a
        cluster (a single leaf of the plant) are dropped in favour of the
        whole plant;
      * near-duplicates keep the larger mask.
    Bounding boxes prune the pairwise tests so this stays fast."""
    def _box(r):
        b = r.get("box")
        if b is None or len(b) != 4:
            ys, xs = np.nonzero(r["mask"])
            b = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1] if len(xs) else [0, 0, 0, 0]
        return [float(v) for v in b]

    def _touch(a, b):
        return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

    inst = [dict(r, _area=int(r["mask"].sum()), _box=_box(r)) for r in instances]
    inst = [r for r in inst if r["_area"] > 0]
    inst.sort(key=lambda r: r["_area"])
    kept: list = []
    for r in inst:
        m, a = r["mask"], r["_area"]
        inside = []            # (kept, overlap px) for kept masks touching r
        for k in kept:
            if not _touch(k["_box"], r["_box"]):
                continue
            ov = int((k["mask"] & m).sum())
            if ov:
                inside.append((k, ov))
        parts = [(k, ov) for k, ov in inside if ov > cover * k["_area"]]
        covered = sum(ov for _, ov in parts) / max(1, a)
        if len(parts) >= 2 and covered > cover:
            # r is a cluster of plants we already have: keep only what is left
            rest = m.copy()
            for k, _ in parts:
                rest &= ~k["mask"]
            if rest.sum() >= max(min_px, 0.15 * a):
                ys, xs = np.nonzero(rest)
                kept.append(dict(r, mask=rest, _area=int(rest.sum()),
                                 _box=[float(xs.min()), float(ys.min()),
                                       float(xs.max() + 1), float(ys.max() + 1)]))
            continue
        # not a cluster: fragments / near-duplicates inside r give way to r
        drop = {id(k) for k, _ in parts}
        kept = [k for k in kept if id(k) not in drop]
        kept.append(r)
    kept.sort(key=lambda r: -r["score"])
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in kept]


def sam_text_instances(st: EditState, prompts, threshold=None) -> list:
    res = engine.concept_instances(st.img, list(prompts),
                                   threshold=threshold, enhance=False)
    if res is None:
        return []
    flat = [r for p in prompts for r in res.get(p, [])]
    return _dedupe(flat, min_px=int(settings.get("min_instance_px", 30))) if flat else []


def sam_exemplar_instances(st: EditState, threshold=None) -> list:
    res = engine.exemplar_instances(st.img, st.pos_boxes, st.neg_boxes,
                                    threshold=threshold)
    return _dedupe(res, min_px=int(settings.get("min_instance_px", 30))) if res else []


def lesion_box_at(st: EditState, x: float, y: float):
    """Tight box of the plant under a click: vegetation component first,
    fallback to a small fixed box."""
    n, lab = cv2.connectedComponents(st.veg.astype(np.uint8), 8)
    cid = lab[int(y), int(x)] if 0 <= int(y) < st.h and 0 <= int(x) < st.w else 0
    if cid > 0:
        ys, xs = np.nonzero(lab == cid)
        return [float(xs.min()), float(ys.min()),
                float(xs.max() + 1), float(ys.max() + 1)]
    r = 20.0
    return [max(0, x - r), max(0, y - r), min(st.w, x + r), min(st.h, y + r)]


def apply_instances(st: EditState, instances: list, cls: int | None,
                    source: str) -> int:
    """Add SAM instances; cls=None -> classify each crop/weed."""
    n = 0
    for r in instances:
        c = cls if cls is not None else clf.classify(st.img, r["mask"])
        if st.add_instance(r["mask"], c, r["score"], source) is not None:
            n += 1
    return n


def run_autolabel(st: EditState) -> str:
    """Full auto pipeline: soil + SAM plant instances + crop/weed classifier."""
    st.sem = np.where(st.veg, CLS_UNLABELED, CLS_SOIL).astype(np.uint8)
    st.inst[:] = 0
    st.meta.clear()
    st.next_id = 1
    st.selected = None
    inst = sam_text_instances(st, settings.get("plant_prompts", ["plant"]),
                              threshold=float(settings["native_threshold"]))
    n = apply_instances(st, inst, None, "autolabel")
    n_split = auto_split_large(st)
    st.compact()
    # any vegetation SAM missed stays visible as 'unlabeled' for review
    clf.maybe_retrain(store(), images(), int(settings["work_res"]))
    return (f"auto-label: {n} plant instances"
            f"{f' (+{n_split} from auto-splitting over-merged ones)' if n_split else ''} "
            f"({'crop/weed classifier active' if clf.model is not None else 'all as crop — classifier not trained yet'})")


def typical_plant_area(st: EditState) -> float:
    """Robust single-plant area. The plain median is pulled up by merged
    clusters and down by leaf fragments, so take the median of the
    instances within 0.5-1.5x of the raw median (the single-plant band)."""
    areas = [m.get("area_px", 0) for m in st.meta.values()
             if m["class"] == CLS_CROP and m.get("area_px")]
    if not areas:
        areas = [m.get("area_px", 0) for m in st.meta.values() if m.get("area_px")]
    return robust_typical_area(areas)


def robust_typical_area(areas) -> float:
    if not areas:
        return 0.0
    med = float(np.median(areas))
    core = [a for a in areas if 0.5 * med <= a <= 1.5 * med]
    return float(np.median(core)) if len(core) >= 3 else med


# Crown-strength filter for automatic split seeds: distance-transform peaks
# weaker than this fraction of the strongest peak are ignored (thin leaves of
# a single plant). Tuned on the 40 OzBarley sample tiles (2026-09-02): 0.35
# keeps ~all two-plant splits of the unfiltered version with ~12% fewer
# fragments; 0.5 starts missing dense clusters.
AUTO_SPLIT_MIN_REL = 0.35


def auto_split_large(st: EditState) -> int:
    """Split instances much larger than the typical plant (SAM often joins
    2-4 neighbouring seedlings). Returns number of NEW instances created."""
    factor = float(settings.get("auto_split_factor", 1.7) or 0)
    if factor <= 0 or len(st.meta) < 4:
        return 0
    typical = typical_plant_area(st)
    if typical <= 0:
        return 0
    created = 0
    for iid in [i for i, m in list(st.meta.items())
                if m.get("area_px", 0) > factor * typical]:
        parts = editing.split_auto(st.inst == iid, typical,
                                   min_px=max(5, int(settings.get("min_instance_px", 30)) // 3),
                                   min_rel=AUTO_SPLIT_MIN_REL)
        if len(parts) > 1:
            created += len(st.split_instance(iid, parts, "auto_split")) - 1
    return created


# ---------------------------------------------------------------------------
# Pages / config / queue
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


class ConfigIn(BaseModel):
    dataset_root: str | None = None
    output_root: str | None = None
    work_res: int | None = None
    native_threshold: float | None = None
    plant_prompts: list[str] | None = None
    exg_thresh: float | None = None
    min_instance_px: int | None = None
    default_crop_type: str | None = None
    show_ids: bool | None = None
    show_boxes: bool | None = None
    save_boxes: bool | None = None
    side_by_side: bool | None = None
    snap_polygon_to_veg: bool | None = None
    gsd_mm_per_px: float | None = None
    clear_gsd: bool | None = None
    auto_split_factor: float | None = None


@app.get("/api/config")
def get_config():
    return settings


@app.post("/api/config")
def set_config(body: ConfigIn):
    d = body.model_dump(exclude_none=True)
    if d.pop("clear_gsd", False):
        settings["gsd_mm_per_px"] = None
    for k, v in d.items():
        settings[k] = v
    config.save_settings(settings)
    return settings


@app.post("/api/browse")
def browse_folder():
    import subprocess
    ps = ("Add-Type -AssemblyName System.Windows.Forms; "
          "$o = New-Object System.Windows.Forms.Form; "
          "$o.TopMost = $true; $o.ShowInTaskbar = $false; "
          "$o.FormBorderStyle = 'None'; $o.Size = New-Object System.Drawing.Size(1,1); "
          "$o.StartPosition = 'CenterScreen'; $o.Show(); $o.Activate(); "
          "$f = New-Object System.Windows.Forms.FolderBrowserDialog; "
          "if ($f.ShowDialog($o) -eq 'OK') { Write-Output $f.SelectedPath } "
          "$o.Close()")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=300)
        return {"path": (out.stdout or "").strip()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.get("/api/queue")
def queue():
    st = store()
    groups = {}
    for key in images():
        parent = str(Path(key).parent)
        g = parent if parent != "." else Path(key).stem.split("_tile")[0]
        groups.setdefault(g, []).append(
            {"key": key, "name": Path(key).name, "status": st.status_of(key)})
    out = [{"group": g, "items": v} for g, v in sorted(groups.items())]
    return {"groups": out, "total": sum(len(v["items"]) for v in out)}


@app.get("/api/version")
def version():
    return {"version": config.APP_VERSION}


@app.get("/api/model/status")
def model_status():
    return {"version": config.APP_VERSION,
            "native_available": engine.available(),
            "native_loaded": engine.proc is not None,
            "native_error": engine.error,
            "classifier": clf.status(),
            "crop_types": settings["crop_types"]}


@app.get("/api/summary")
def summary():
    """Dataset-level labeling summary (all reviewed + auto images)."""
    return store().summary()


# ---------------------------------------------------------------------------
# Open / actions
# ---------------------------------------------------------------------------
class KeyIn(BaseModel):
    key: str


def _response(st: EditState, msg: str) -> dict:
    return {"overlay_b64": jpg_b64(st.overlay()), "msg": msg,
            **st.counts(), "selected": st.selected_info(),
            "veg_pct": round(100.0 * float(st.veg.mean()), 2)}


@app.post("/api/open")
def open_image(body: KeyIn):
    try:
        st = state_of(body.key)
    except KeyError:
        return JSONResponse({"error": "unknown key"}, 404)
    return {"key": st.key, "w": st.w, "h": st.h,
            "img_b64": jpg_b64(st.img),
            "crop_type": st.crop_type,
            "status": store().status_of(st.key),
            "record": store().records().get(st.key),
            **_response(st, "")}


class ActIn(BaseModel):
    key: str
    action: str
    x: float | None = None
    y: float | None = None
    box: list | None = None          # [x0,y0,x1,y1]
    cls: int | None = None           # 2=crop 3=weed (1=soil for brush)
    negative: bool = False
    prompt: str | None = None
    threshold: float | None = None
    path: list | None = None         # polyline / polygon [[x,y],...]
    points: list | None = None       # seed points for split_points
    radius: float | None = None
    crop_type: str | None = None
    iid: int | None = None           # explicit instance id (select/merge)
    to_selected: bool = False        # brush adds to the selected instance
    snap: bool | None = None         # polygon: keep vegetation pixels only
    steal: bool = False              # brush/polygon may take pixels of others
    n: int | None = None             # split_auto: number of plants expected


_act_lock = threading.Lock()


@app.post("/api/act")
def act(body: ActIn):
    with _act_lock:   # actions mutate shared state — strictly one at a time
        return _act(body)


def _cls_name(c):
    return "crop" if c == CLS_CROP else "weed"


def _act(body: ActIn):
    try:
        st = state_of(body.key)
    except KeyError:
        return JSONResponse({"error": "unknown key"}, 404)
    a = body.action
    msg = ""
    min_px = int(settings.get("min_instance_px", 30))

    # ------------------------------------------------------------ selection
    if a == "select":
        iid = body.iid if body.iid is not None else st.instance_at(body.x, body.y)
        if iid and iid in st.meta:
            st.selected = iid
            m = st.meta[iid]
            msg = (f"selected instance {iid} ({_cls_name(m['class'])}, "
                   f"{m.get('area_px', 0)} px) — brush/polygon now ADD to it; "
                   f"✂ tools split it; Delete removes it")
        else:
            st.selected = None
            msg = "selection cleared"
        return _response(st, msg)   # no dirty flag, no undo

    if a == "deselect":
        st.selected = None
        return _response(st, "selection cleared")

    # -------------------------------------------------------------- SAM
    if a == "exemplar":
        # click or box on ONE example plant -> find all similar
        box = body.box or lesion_box_at(st, body.x, body.y)
        (st.neg_boxes if body.negative else st.pos_boxes).append(box)
        if not st.pos_boxes:
            msg = "negative example noted — now click a real plant"
        else:
            if not engine.available():
                return JSONResponse({"error": "SAM3.1 engine unavailable "
                                     f"({engine.error})"}, 503)
            st.push_undo()
            # exemplar replaces previous exemplar-sourced instances of this class
            for iid in [i for i, m in list(st.meta.items())
                        if m["source"] == f"exemplar:{body.cls}"]:
                st.remove_instance(iid)
            inst = sam_exemplar_instances(st, threshold=body.threshold)
            n = apply_instances(st, inst, body.cls, f"exemplar:{body.cls}")
            msg = (f"⭐ {n} similar plants labeled as {_cls_name(body.cls)} "
                   f"({len(st.pos_boxes)}+ {len(st.neg_boxes)}- examples)")
    elif a == "exemplar_reset":
        st.pos_boxes, st.neg_boxes = [], []
        msg = "exemplar examples cleared"
    elif a == "text_prompt":
        if not body.prompt:
            return JSONResponse({"error": "empty prompt"}, 400)
        if not engine.available():
            return JSONResponse({"error": f"SAM3.1 unavailable ({engine.error})"}, 503)
        st.push_undo()
        inst = sam_text_instances(st, [body.prompt], threshold=body.threshold)
        n = apply_instances(st, inst, body.cls, f"prompt:{body.prompt}")
        msg = f"'{body.prompt}': {n} new instances added"
    elif a == "autolabel":
        if not engine.available():
            return JSONResponse({"error": f"SAM3.1 unavailable ({engine.error})"}, 503)
        st.push_undo()
        msg = run_autolabel(st)

    # ----------------------------------------------------- create instances
    elif a == "add_one":
        # segment just the plant under the click as ONE instance
        box = lesion_box_at(st, body.x, body.y)
        st.push_undo()
        m = np.zeros((st.h, st.w), bool)
        x0, y0, x1, y1 = (int(v) for v in box)
        m[y0:y1, x0:x1] = st.veg[y0:y1, x0:x1]
        iid = st.add_instance(m, body.cls or CLS_CROP, 1.0, "manual")
        if iid:
            st.selected = iid
            msg = f"plant added as instance {iid}"
        else:
            msg = "nothing under that click (no vegetation)"
    elif a == "polygon":
        # user clicked the corners of a plant -> new instance (or add to selected)
        pts = body.path or []
        if len(pts) < 3:
            return JSONResponse({"error": "need at least 3 corners"}, 400)
        poly = editing.polygon_mask((st.h, st.w), pts)
        snap = settings.get("snap_polygon_to_veg", True) if body.snap is None else body.snap
        mask = poly
        if snap:
            g = cv2.dilate(st.veg.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            snapped = poly & g
            # if the polygon holds (almost) no vegetation, keep the raw polygon
            mask = snapped if snapped.sum() >= max(10, 0.05 * poly.sum()) else poly
        st.push_undo()
        if body.to_selected and st.selected in st.meta:
            n = st.grow_instance(st.selected, mask, steal=body.steal)
            msg = f"+{n} px added to instance {st.selected}"
        else:
            if body.steal:
                # take the polygon area away from whoever owns it
                owners = [int(i) for i in np.unique(st.inst[mask]) if i]
                st.inst[mask] = 0
                for o in owners:
                    st.refresh_meta(o)
            iid = st.add_instance(mask, body.cls or CLS_CROP, 1.0, "polygon",
                                  gate=False)
            if iid:
                st.selected = iid
                msg = (f"polygon → new {_cls_name(body.cls or CLS_CROP)} instance {iid} "
                       f"({int((st.inst == iid).sum())} px"
                       f"{', snapped to vegetation' if snap and mask is not poly else ''})")
            else:
                msg = ("polygon too small or fully inside existing instances "
                       "(select an instance first to add to it, or use ⌫ eraser)")

    # ------------------------------------------------------ brush / eraser
    elif a == "brush":
        r = int(body.radius or 10)
        stamp = editing.paint_stamp((st.h, st.w), body.path, r)
        st.push_undo()
        cls = body.cls if body.cls is not None else CLS_CROP
        if cls == CLS_SOIL:
            # eraser: remove pixels (from the selected instance only, if one
            # is selected — precise cleanup; otherwise from anything)
            if st.selected in st.meta:
                n = st.shrink_instance(st.selected, stamp)
                msg = f"erased {n} px from instance {st.selected}"
            else:
                st.sem[stamp] = CLS_SOIL
                touched = [int(i) for i in np.unique(st.inst[stamp]) if i]
                st.inst[stamp] = 0
                for iid in touched:
                    st.refresh_meta(iid)
                msg = "erased to soil"
        elif st.selected in st.meta and body.to_selected:
            n = st.grow_instance(st.selected, stamp, steal=body.steal)
            msg = f"+{n} px painted into instance {st.selected}"
        else:
            # no instance selected: paint semantic class on free pixels only
            free = stamp & (st.inst == 0)
            st.sem[free] = cls
            msg = (f"painted {_cls_name(cls)} (semantic only — select an "
                   f"instance to paint INTO it, or use polygon to create one)")

    # ------------------------------------------------------------- splits
    elif a in ("split_line", "split_points", "split_box", "split_auto"):
        iid = st.selected if st.selected in st.meta else None
        if iid is None:
            # infer from the geometry: instance under the first point / box centre
            if a == "split_auto":
                iid = st.instance_at(body.x, body.y) or None
            elif a == "split_box" and body.box:
                cx, cy = (body.box[0] + body.box[2]) / 2, (body.box[1] + body.box[3]) / 2
                iid = st.instance_at(cx, cy) or None
            else:
                pts = body.points or body.path or []
                for x, y in pts:
                    iid = st.instance_at(x, y)
                    if iid:
                        break
                iid = iid or None
        if iid is None:
            return JSONResponse({"error": "select (or draw on) an instance to split"}, 400)
        whole = st.inst == iid
        if a == "split_line":
            parts = editing.split_by_line(whole, body.path or [],
                                          width=max(2, int(body.radius or 3)),
                                          min_px=max(5, min_px // 3))
        elif a == "split_points":
            parts = editing.split_by_points(whole, body.points or [],
                                            min_px=max(5, min_px // 3))
        elif a == "split_box":
            parts = editing.split_by_box(whole, body.box or [0, 0, 0, 0],
                                         min_px=max(5, min_px // 3))
        else:
            typical = typical_plant_area(st) or float(whole.sum()) / 2
            parts = editing.split_auto(whole, typical, min_px=max(5, min_px // 3),
                                       n_parts=body.n if body.n and body.n > 1 else None,
                                       min_rel=0.0 if body.n else AUTO_SPLIT_MIN_REL)
        if len(parts) < 2:
            hint = {"split_line": "the line must cross the plant completely",
                    "split_points": "click one seed INSIDE each plant (2+)",
                    "split_box": "the box must cover part of the plant, not all",
                    "split_auto": "no separate crowns found — use ✂• seeds instead"}[a]
            return _response(st, f"nothing to split — {hint}")
        st.push_undo()
        ids = st.split_instance(iid, parts, source=a)
        st.selected = None
        msg = (f"✂ instance {iid} split into {len(ids)}: {ids} "
               f"(instances after it renumbered)")

    # --------------------------------------------------------------- merge
    elif a == "merge":
        other = st.instance_at(body.x, body.y) if body.iid is None else body.iid
        if st.selected in st.meta and other and other != st.selected:
            st.push_undo()
            st.merge_instances(st.selected, other)
            msg = f"merged instance {other} into {st.selected}"
        elif other:
            st.selected = other
            msg = f"instance {other} selected — now click the instance to merge into it"
        else:
            msg = "no instance under that click"

    # ------------------------------------------------------ delete / class
    elif a == "delete":
        iid = body.iid if body.iid is not None else st.instance_at(body.x, body.y)
        if not iid and st.selected in st.meta and body.x is None:
            iid = st.selected
        if iid and iid in st.meta:
            st.push_undo()
            st.remove_instance(iid)
            msg = f"instance {iid} deleted"
        else:
            msg = "no instance under that click"
    elif a == "reclass":
        iid = body.iid if body.iid is not None else st.instance_at(body.x, body.y)
        if not iid and st.selected in st.meta and body.x is None:
            iid = st.selected
        if iid and iid in st.meta and body.cls in (CLS_CROP, CLS_WEED):
            st.push_undo()
            st.reclass_instance(iid, body.cls)
            msg = f"instance {iid} → {_cls_name(body.cls)}"
        else:
            msg = "no instance under that click"

    # --------------------------------------------------------------- misc
    elif a == "soil_only":
        st.push_undo()
        st.sem = np.where(st.veg, CLS_UNLABELED, CLS_SOIL).astype(np.uint8)
        st.inst[:] = 0
        st.meta.clear()
        st.selected = None
        msg = "reset to auto-soil"
    elif a == "set_crop_type":
        st.crop_type = body.crop_type or st.crop_type
        msg = f"crop type: {st.crop_type}"
    elif a == "undo":
        msg = "undone" if st.do_undo() else "nothing to undo"
    elif a == "redo":
        msg = "redone" if st.do_redo() else "nothing to redo"
    elif a == "render":
        return _response(st, "")       # re-render after a view option change
    else:
        return JSONResponse({"error": f"unknown action {a}"}, 400)

    st.compact()
    st.dirty = True
    return _response(st, msg)


class SaveIn(BaseModel):
    key: str
    crop_type: str | None = None
    status: str = "reviewed"


@app.post("/api/save")
def save(body: SaveIn):
    st = state_of(body.key)
    if body.crop_type:
        st.crop_type = body.crop_type
    st.compact()
    rec = store().save(st.key, st.img_path, st.sem, st.inst,
                       st.instances_meta(), st.crop_type,
                       "sam3.1+human", status=body.status,
                       options=save_options())
    st.dirty = False
    clf.maybe_retrain(store(), images(), int(settings["work_res"]))
    return {"ok": True, "record": rec, "classifier": clf.status(),
            "summary": store().summary()}


@app.post("/api/skip")
def skip(body: KeyIn):
    store().mark_skipped(body.key)
    return {"ok": True}


@app.get("/api/instances")
def list_instances(key: str):
    try:
        st = state_of(key)
    except KeyError:
        return JSONResponse({"error": "unknown key"}, 404)
    return {"instances": sorted(st.instances_meta(), key=lambda m: m["id"]),
            "selected": st.selected}


# ---------------------------------------------------------------------------
# Batch auto-label
# ---------------------------------------------------------------------------
_batch = {"running": False, "done": 0, "total": 0, "current": "",
          "stop": False, "errors": []}
_batch_lock = threading.Lock()


def _batch_worker(keys):
    imgs = images()
    for i, key in enumerate(keys):
        with _batch_lock:
            if _batch["stop"]:
                break
            _batch["current"] = key
        try:
            st = EditState(key, imgs[key])       # fresh, not the UI state
            run_autolabel(st)
            store().save(key, st.img_path, st.sem, st.inst,
                         st.instances_meta(),
                         settings.get("default_crop_type", "barley"),
                         "sam3.1-auto", status="auto", options=save_options())
        except Exception as e:
            with _batch_lock:
                _batch["errors"].append(f"{key}: {e}")
        with _batch_lock:
            _batch["done"] = i + 1
    with _batch_lock:
        _batch["running"] = False
        _batch["current"] = ""


@app.post("/api/autolabel_all")
def autolabel_all():
    with _batch_lock:
        if _batch["running"]:
            return {"started": False, **_batch}
        st = store()
        keys = [k for k in images() if st.status_of(k) == "pending"]
        _batch.update(running=True, done=0, total=len(keys), current="",
                      stop=False, errors=[])
    threading.Thread(target=_batch_worker, args=(keys,), daemon=True).start()
    return {"started": True, **_batch}


@app.get("/api/autolabel_all/status")
def autolabel_all_status():
    with _batch_lock:
        return dict(_batch)


@app.post("/api/autolabel_all/stop")
def autolabel_all_stop():
    with _batch_lock:
        _batch["stop"] = True
    return {"ok": True}
