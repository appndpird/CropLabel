# CropLabel

Agentic labeling tool for crop-plot images (UAV tiles): **instance + semantic
segmentation** of crop plants, weeds and soil with SAM3.1 assistance. One crop
species per image (barley / wheat / canola / oats / other), weeds possible,
soil labeled fully automatically.

Built from the RetinaLabel architecture (FastAPI + browser canvas + SAM3.1
native GPU engine with text and visual-exemplar prompts).

## The agentic loop

1. **Soil is free** — every image opens with soil already labeled (Excess
   Green index: living vegetation vs everything else).
2. **⭐ label a couple of plants** — click ONE example crop plant; SAM3.1
   finds every similar plant in the image and labels each as a separate
   instance. Same for weeds (select the Weed class, click a weed).
   `Shift+click` = negative example. Each extra example refines the search.
3. **🤖 tell it what to find** — the free-text box runs any description
   ("broad leaf plant", "thin grass seedling") as a SAM3 concept prompt for
   the selected class.
4. **Save** — labels are written at the ORIGINAL image size with the SAME
   filename as the input. Every save also trains the **crop-vs-weed
   classifier** (color/shape/texture features per instance), so batch
   auto-labels keep improving as you review.
5. **⚡ Label ALL remaining** — after a couple of reviewed images, the rest of
   the folder is labeled automatically: soil + SAM3 plant instances + the
   learned classifier deciding crop vs weed per instance.

## Classes

Semantic mask values: `0` unlabeled · `1` soil · `2` crop · `3` weed.
The actual crop species is stored per image (`crop_type` in the JSON).

## Outputs (per image, mirrored under the output root)

| File | Content |
|------|---------|
| `<name>.png` | semantic index mask, **same filename & size as the input** |
| `<name>_instances.png` | uint16 instance-ID map (instance segmentation) |
| `<name>_color.png`, `<name>_overlay.png` | visual checks |
| `<name>.json` | crop type, per-instance records (id, class, score, bbox, centroid, area) + **summary statistics** (see below) |
| `<name>_boxes.json` / `_boxes.txt` / `_boxes.png` | **optional** (tick *Save bbox files*): bounding boxes as JSON (xyxy px), YOLO txt (`cls cx cy w h` normalized, 0=crop 1=weed) and a visual |
| `records.json` | queue status store |
| `summary.csv` | one row per labeled image with every summary statistic — open in Excel |

### Summary statistics (per image, in `<name>.json`, `records.json`, `summary.csv`)

`n_instances`, `n_crop_instances`, `n_weed_instances`, `crop_pct`, `weed_pct`,
`soil_pct`, `unlabeled_pct`, `greenness_ratio` (fraction of ExG-vegetation
pixels), `mean_exg`, `mean_exg_crop`, `labeled_veg_fraction`, `mean/median
crop area (px)`. With the **GSD (mm/px)** set in ⚙ Setup also:
`image_area_m2`, `plants_per_m2`, `weeds_per_m2`, `crop_cover_m2`,
`mean_crop_area_cm2`. The 📊 Summary panel shows the current image and the
dataset-wide aggregates.

## Run

`CropLabel.bat` (or `python run_app.py`) → http://127.0.0.1:8323

Needs the same Python environment as RetinaLabel (torch+cu128, sam3 package,
opencv, fastapi). The SAM3.1 checkpoint lives in `models/sam3.1_multiplex.pt`.

## Review workflow

1. ✨ **Auto-label** (SAM3.1 + crop/weed classifier) — or ⚡ for the whole folder.
2. Review: tick **Instance numbers** / **Bounding boxes** / **Side by side**
   (original next to annotated) in the option bar.
3. Fix with the manual tools below — select an instance (➚ or `Alt+click`),
   then brush pixels into it, split it, merge it, delete it or reclassify it.
4. 💾 **Save** — masks, JSON, optional bbox files and the summary statistics.

## Tools

| Group | Tool | Key | Action |
|-------|------|-----|--------|
| SAM | ⭐ find similar | 1 | click/box one example plant → all similar labeled as selected class; Shift = negative example |
| SAM | ✚ add one | 2 | add just the plant under the click as one instance |
| Create | ⬠ polygon | 3 | click the corners of a plant, double-click / Enter to finish → **new instance**. Backspace = remove corner, Esc = cancel. With *Snap polygon to vegetation* on, only green pixels inside the polygon are kept (accurate outline from a rough polygon). A polygon that **touches** the selected instance is added to it (missed leaf); a polygon elsewhere always makes a **new instance** (hold Shift while closing to force adding). |
| Create | 🖌 brush | 4 | with an instance **selected**: paints missing pixels **into that instance**. Without selection: semantic paint only. `Ctrl` while painting also takes pixels from neighbouring instances. `[` / `]` change size. |
| Create | ⌫ eraser | 5 | removes pixels from the selected instance (nothing selected: erases any label to soil) |
| Fix | ➚ select | 6 | select an instance (yellow outline). `Delete` removes it, `c` / `w` reclassify it, the floating panel offers the same. `Alt+click` selects with **any** tool. |
| Fix | ✂ split by line | 7 | drag a line across a merged plant → cut into separate instances along the line (start and end outside the plant) |
| Fix | ✂• split by seeds | 8 | click one seed **inside each plant** of a merged instance, then Enter / double-click → watershed splits it into one instance per seed (follows the natural neck between plants) |
| Fix | ✂▭ split by box | 9 | drag a box over the part that should be its own plant |
| Fix | ✂⬠ split by polygon | P | click the corners around **one** plant of a merged instance (like the polygon tool), Enter / double-click / first corner to close → that part becomes its own instance, the rest keeps its number |
| Fix | ✂A auto split | A | click an over-merged instance → split automatically at the plant crowns; Shift+click to say how many plants it holds. ✨ Auto-label already applies this to every instance larger than *Auto-split ×* the typical single-plant area (Setup, default 1.7; 0 = off) |
| Fix | ⛓ merge | 0 | click the first instance, then the second → one instance |
| Fix | 🗑 delete | D | remove the clicked instance |
| Fix | 🏷 reclassify | R | clicked instance → selected class (crop ↔ weed) |
| View | ✋ hand | H | left-drag moves the image (right-drag, middle-drag or Space+drag move it with **any** tool); `−` / `+` / `⤢ Fit` / `1:1` buttons, arrow keys nudge, `F` fits, wheel zooms at the cursor. In *Side by side* mode each panel zooms and moves on its own (wheel / drag inside it); tick **🔗 Link views** to lock them together |
| keys | | | `c` crop, `w` weed, `Ctrl+Z/Y` undo/redo, `Ctrl+S` save, `Esc` cancel / deselect |

Every action is undoable (25 steps). Instance numbers are always sequential:
splitting instance 13 into three gives 13, 14, 15 and shifts the later ones;
delete/merge close the gap. The label files always contain exactly the
instances you see, with the same numbers.

## Dataset

`dataset/` holds the starter subset: 40 random tiles (1600×1600) from 40
different plots of `Ozbarley2025\Tiles_sample\tiles`. Point the Setup panel at
any other folder to label it (images are found recursively).
