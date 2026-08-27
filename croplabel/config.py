"""CropLabel configuration and persistent app settings."""
import json
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
SETTINGS_PATH = APP_DIR / "settings.json"

DEFAULTS = {
    "dataset_root": str(APP_DIR / "dataset"),
    "output_root": str(APP_DIR / "labels"),
    "work_res": 1024,           # SAM3 working resolution (labels are always
                                # saved at the ORIGINAL image size)
    "sam3_ckpt": str(APP_DIR / "models" / "sam3.1_multiplex.pt"),
    "native_threshold": 0.4,    # SAM3.1 concept confidence threshold
    "plant_prompts": ["plant", "seedling"],  # auto-label text prompts
    "exg_thresh": 0.06,         # ExG vegetation threshold (soil = below)
    "min_instance_px": 30,      # drop instances smaller than this (@work_res)
    "veg_gate": True,           # clip SAM instances to the vegetation mask
    "crop_types": ["barley", "wheat", "canola", "oats", "other"],
    "default_crop_type": "barley",
    # ---- review / display options (toggled from the toolbar, persisted)
    "show_ids": False,          # draw the instance number on top of each plant
    "show_boxes": False,        # draw per-instance bounding boxes
    "save_boxes": False,        # also write <name>_boxes.json / .txt (YOLO) / .png
    "side_by_side": False,      # original image next to the annotated one
    "snap_polygon_to_veg": True,  # polygon tool keeps only vegetation pixels
    # ---- summary statistics
    "gsd_mm_per_px": None,      # ground sampling distance of the INPUT images
                                # (mm per pixel). Needed for plants per m^2.
    # semantic classes (fixed): 0=unlabeled, 1=soil, 2=crop, 3=weed
    "classes": [
        {"id": 1, "name": "soil",  "color": "#8a6a4b"},
        {"id": 2, "name": "crop",  "color": "#37c837"},
        {"id": 3, "name": "weed",  "color": "#e63c3c"},
    ],
}

CLS_UNLABELED = 0
CLS_SOIL = 1
CLS_CROP = 2
CLS_WEED = 3

CLASS_COLORS_BGR = {
    CLS_UNLABELED: (0, 0, 0),
    CLS_SOIL: (75, 106, 138),    # brown (BGR)
    CLS_CROP: (55, 200, 55),     # green
    CLS_WEED: (60, 60, 230),     # red
}
CLASS_NAMES = {CLS_UNLABELED: "unlabeled", CLS_SOIL: "soil",
               CLS_CROP: "crop", CLS_WEED: "weed"}


def load_settings() -> dict:
    s = dict(DEFAULTS)
    if SETTINGS_PATH.exists():
        try:
            s.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    return s


def save_settings(s: dict) -> None:
    keep = {k: s[k] for k in DEFAULTS if k in s}
    SETTINGS_PATH.write_text(json.dumps(keep, indent=2), encoding="utf-8")
