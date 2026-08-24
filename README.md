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
| `<name>.json` | crop type, per-instance records (id, class, score, bbox, area) |
| `records.json` | queue status store |

## Run

`CropLabel.bat` (or `python run_app.py`) → http://127.0.0.1:8323

Needs the same Python environment as RetinaLabel (torch+cu128, sam3 package,
opencv, fastapi). The SAM3.1 checkpoint lives in `models/sam3.1_multiplex.pt`.

## Tools

| Tool | Key | Action |
|------|-----|--------|
| ⭐ find similar | 1 | click/box one example plant → all similar labeled as selected class; Shift = negative example |
| ✚ add one | 2 | add just the plant under the click as one instance |
| 🗑 delete | 3 | remove the clicked instance |
| 🏷 reclassify | 4 | clicked instance → selected class (crop ↔ weed) |
| 🖌 brush / ⌫ eraser | 5/6 | manual semantic paint / erase to soil |
| keys | | `c` crop, `w` weed, `Ctrl+Z/Y` undo/redo, `Ctrl+S` save, wheel zoom, right-drag pan |

## Dataset

`dataset/` holds the starter subset: 40 random tiles (1600×1600) from 40
different plots of `Ozbarley2025\Tiles_sample\tiles`. Point the Setup panel at
any other folder to label it (images are found recursively).
