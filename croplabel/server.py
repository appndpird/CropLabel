"""CropLabel FastAPI server — agentic crop/weed/soil labeling with SAM3.1.

Design: the server owns the editing state (semantic mask + instance map per
open image). The browser sends ACTIONS (exemplar click, delete, brush, ...)
and receives a freshly rendered overlay. All segmentation, gating and
learning happens here in Python.
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

from . import config
from .config import CLS_CROP, CLS_SOIL, CLS_UNLABELED, CLS_WEED
from .classifier import CropWeedClassifier
from .labelstore import LabelStore, colorize
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
        self.meta = {}          # id -> {class, score, source}
        self.next_id = 1
        self.crop_type = settings.get("default_crop_type", "barley")
        self.pos_boxes, self.neg_boxes = [], []   # exemplar memory (this img)
        self.undo, self.redo = [], []
        self.dirty = False

    # ------------------------------------------------------------- history
    def push_undo(self):
        self.undo.append((self.sem.copy(), self.inst.copy(),
                          dict(self.meta), self.next_id))
        if len(self.undo) > 15:
            self.undo.pop(0)
        self.redo = []

    def do_undo(self):
        if not self.undo:
            return False
        self.redo.append((self.sem, self.inst, self.meta, self.next_id))
        self.sem, self.inst, self.meta, self.next_id = self.undo.pop()
        return True

    def do_redo(self):
        if not self.redo:
            return False
        self.undo.append((self.sem, self.inst, self.meta, self.next_id))
        self.sem, self.inst, self.meta, self.next_id = self.redo.pop()
        return True

    # ----------------------------------------------------------- instances
    def add_instance(self, mask: np.ndarray, cls: int, score: float,
                     source: str) -> int | None:
        if settings.get("veg_gate", True):
            gate = cv2.dilate(self.veg.astype(np.uint8),
                              np.ones((5, 5), np.uint8)) > 0
            mask = mask & gate
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
        ys, xs = np.nonzero(mask)
        self.meta[iid] = {"id": iid, "class": int(cls),
                          "score": round(float(score), 3), "source": source,
                          "bbox": [int(xs.min()), int(ys.min()),
                                   int(xs.max() + 1), int(ys.max() + 1)]}
        return iid

    def remove_instance(self, iid: int):
        m = self.inst == iid
        self.inst[m] = 0
        self.sem[m] = np.where(self.veg[m], CLS_UNLABELED, CLS_SOIL)
        self.meta.pop(iid, None)

    def reclass_instance(self, iid: int, cls: int):
        m = self.inst == iid
        self.sem[m] = cls
        if iid in self.meta:
            self.meta[iid]["class"] = int(cls)

    def instances_meta(self) -> list:
        return [dict(v) for v in self.meta.values()]

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
                st.meta[int(m["id"])] = {"id": int(m["id"]),
                                         "class": int(m["class"]),
                                         "score": m.get("score", 1.0),
                                         "source": m.get("source", "saved"),
                                         "bbox": m.get("bbox")}
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
def _dedupe(instances: list) -> list:
    """Drop instances whose mask mostly overlaps a higher-scoring one."""
    inst = sorted(instances, key=lambda r: -r["score"])
    kept, used = [], np.zeros(inst[0]["mask"].shape, bool) if inst else None
    for r in inst:
        inter = (r["mask"] & used).sum()
        if inter / max(1, r["mask"].sum()) > 0.6:
            continue
        kept.append(r)
        used |= r["mask"]
    return kept


def sam_text_instances(st: EditState, prompts, threshold=None) -> list:
    res = engine.concept_instances(st.img, list(prompts),
                                   threshold=threshold, enhance=False)
    if res is None:
        return []
    flat = [r for p in prompts for r in res.get(p, [])]
    return _dedupe(flat) if flat else []


def sam_exemplar_instances(st: EditState, threshold=None) -> list:
    res = engine.exemplar_instances(st.img, st.pos_boxes, st.neg_boxes,
                                    threshold=threshold)
    return _dedupe(res) if res else []


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
    inst = sam_text_instances(st, settings.get("plant_prompts", ["plant"]),
                              threshold=float(settings["native_threshold"]))
    n = apply_instances(st, inst, None, "autolabel")
    # any vegetation SAM missed stays visible as 'unlabeled' for review
    clf.maybe_retrain(store(), images(), int(settings["work_res"]))
    return (f"auto-label: {n} plant instances "
            f"({'crop/weed classifier active' if clf.model is not None else 'all as crop — classifier not trained yet'})")


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


@app.get("/api/config")
def get_config():
    return settings


@app.post("/api/config")
def set_config(body: ConfigIn):
    for k, v in body.model_dump(exclude_none=True).items():
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


@app.get("/api/model/status")
def model_status():
    return {"native_available": engine.available(),
            "native_loaded": engine.proc is not None,
            "native_error": engine.error,
            "classifier": clf.status(),
            "crop_types": settings["crop_types"]}


# ---------------------------------------------------------------------------
# Open / actions
# ---------------------------------------------------------------------------
class KeyIn(BaseModel):
    key: str


@app.post("/api/open")
def open_image(body: KeyIn):
    try:
        st = state_of(body.key)
    except KeyError:
        return JSONResponse({"error": "unknown key"}, 404)
    return {"key": st.key, "w": st.w, "h": st.h,
            "img_b64": jpg_b64(st.img),
            "overlay_b64": jpg_b64(st.overlay()),
            "crop_type": st.crop_type,
            "status": store().status_of(st.key),
            "n_instances": len(st.meta),
            "veg_pct": round(100.0 * float(st.veg.mean()), 2)}


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
    path: list | None = None         # brush polyline [[x,y],...]
    radius: float | None = None
    crop_type: str | None = None


_act_lock = threading.Lock()


@app.post("/api/act")
def act(body: ActIn):
    with _act_lock:   # actions mutate shared state — strictly one at a time
        return _act(body)


def _act(body: ActIn):
    try:
        st = state_of(body.key)
    except KeyError:
        return JSONResponse({"error": "unknown key"}, 404)
    a = body.action
    msg = ""

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
            msg = (f"⭐ {n} similar plants labeled as "
                   f"{'crop' if body.cls == CLS_CROP else 'weed'} "
                   f"({len(st.pos_boxes)}+ {len(st.neg_boxes)}- examples)")
    elif a == "exemplar_reset":
        st.pos_boxes, st.neg_boxes = [], []
        msg = "exemplar examples cleared"
    elif a == "add_one":
        # segment just the plant under the click as ONE instance
        box = lesion_box_at(st, body.x, body.y)
        st.push_undo()
        m = np.zeros((st.h, st.w), bool)
        x0, y0, x1, y1 = (int(v) for v in box)
        sub = st.veg[y0:y1, x0:x1]
        m[y0:y1, x0:x1] = sub
        iid = st.add_instance(m, body.cls or CLS_CROP, 1.0, "manual")
        msg = "plant added" if iid else "nothing under that click (no vegetation)"
    elif a == "delete":
        iid = int(st.inst[int(body.y), int(body.x)])
        if iid:
            st.push_undo()
            st.remove_instance(iid)
            msg = f"instance {iid} deleted"
        else:
            msg = "no instance under that click"
    elif a == "reclass":
        iid = int(st.inst[int(body.y), int(body.x)])
        if iid and body.cls in (CLS_CROP, CLS_WEED):
            st.push_undo()
            st.reclass_instance(iid, body.cls)
            msg = f"instance {iid} → {'crop' if body.cls == CLS_CROP else 'weed'}"
        else:
            msg = "no instance under that click"
    elif a == "brush":
        st.push_undo()
        r = int(body.radius or 10)
        cls = body.cls if body.cls is not None else CLS_CROP
        stamp = np.zeros((st.h, st.w), np.uint8)
        pts = np.array(body.path or [], np.int32)
        for i in range(len(pts)):
            cv2.circle(stamp, tuple(pts[i]), r, 1, -1)
            if i:
                cv2.line(stamp, tuple(pts[i - 1]), tuple(pts[i]), 1, r * 2)
        m = stamp > 0
        st.sem[m] = cls
        if cls == CLS_SOIL:            # erasing removes instance ownership
            st.inst[m] = 0
            gone = set(st.meta) - set(np.unique(st.inst))
            for iid in gone:
                st.meta.pop(iid, None)
        msg = "painted"
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
    elif a == "soil_only":
        st.push_undo()
        st.sem = np.where(st.veg, CLS_UNLABELED, CLS_SOIL).astype(np.uint8)
        st.inst[:] = 0
        st.meta.clear()
        msg = "reset to auto-soil"
    elif a == "set_crop_type":
        st.crop_type = body.crop_type or st.crop_type
        msg = f"crop type: {st.crop_type}"
    elif a == "undo":
        msg = "undone" if st.do_undo() else "nothing to undo"
    elif a == "redo":
        msg = "redone" if st.do_redo() else "nothing to redo"
    else:
        return JSONResponse({"error": f"unknown action {a}"}, 400)

    st.dirty = True
    return {"overlay_b64": jpg_b64(st.overlay()), "msg": msg,
            "n_instances": len(st.meta),
            "n_crop": sum(1 for m in st.meta.values() if m["class"] == CLS_CROP),
            "n_weed": sum(1 for m in st.meta.values() if m["class"] == CLS_WEED)}


class SaveIn(BaseModel):
    key: str
    crop_type: str | None = None
    status: str = "reviewed"


@app.post("/api/save")
def save(body: SaveIn):
    st = state_of(body.key)
    if body.crop_type:
        st.crop_type = body.crop_type
    rec = store().save(st.key, st.img_path, st.sem, st.inst,
                       st.instances_meta(), st.crop_type,
                       "sam3.1+human", status=body.status)
    st.dirty = False
    clf.maybe_retrain(store(), images(), int(settings["work_res"]))
    return {"ok": True, "record": rec, "classifier": clf.status()}


@app.post("/api/skip")
def skip(body: KeyIn):
    store().mark_skipped(body.key)
    return {"ok": True}


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
                         "sam3.1-auto", status="auto")
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
