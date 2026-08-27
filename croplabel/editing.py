"""Manual instance-editing primitives for CropLabel.

Pure numpy/OpenCV functions operating on (sem, inst) label arrays. They
return new masks / component lists; the EditState in server.py applies them
(so undo/redo and metadata bookkeeping stay in one place).

Tools implemented here:
  polygon_mask        corners clicked by the user -> filled polygon
  split_by_line       polyline cut across a merged instance -> N parts
  split_by_points     one seed click per plant -> marker watershed -> N parts
  split_by_box        box drawn inside an instance -> that part is separated
  paint_stamp         brush polyline -> mask (used for add-to-instance / erase)
"""
import cv2
import numpy as np


def paint_stamp(shape, path, radius: int) -> np.ndarray:
    """Bool mask of a brush stroke (polyline with round caps)."""
    stamp = np.zeros(shape, np.uint8)
    pts = np.array(path or [], np.int32).reshape(-1, 2)
    r = max(1, int(radius))
    for i in range(len(pts)):
        cv2.circle(stamp, tuple(pts[i]), r, 1, -1)
        if i:
            cv2.line(stamp, tuple(pts[i - 1]), tuple(pts[i]), 1, r * 2)
    return stamp > 0


def polygon_mask(shape, points) -> np.ndarray:
    """Bool mask of the polygon whose corners the user clicked."""
    m = np.zeros(shape, np.uint8)
    pts = np.array(points or [], np.int32).reshape(-1, 2)
    if len(pts) >= 3:
        cv2.fillPoly(m, [pts], 1)
    return m > 0


def _components(mask: np.ndarray, min_px: int):
    """Connected components of a bool mask, largest first, tiny ones dropped."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    comps = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            comps.append(lab == i)
    comps.sort(key=lambda c: -int(c.sum()))
    return comps


def _grow_into(parts: list, gap: np.ndarray, iters: int = 50) -> list:
    """Assign 'gap' pixels (e.g. the cut line) to the nearest part so a split
    leaves no unlabeled seam. Iterative competitive dilation."""
    if not parts or not gap.any():
        return parts
    lab = np.zeros(gap.shape, np.int32)
    for i, p in enumerate(parts, 1):
        lab[p] = i
    k = np.ones((3, 3), np.uint8)
    todo = gap & (lab == 0)
    for _ in range(iters):
        if not todo.any():
            break
        grown = cv2.dilate(lab.astype(np.float32), k)   # max-filter of labels
        # only accept a label where exactly-one neighbour label exists is
        # overkill; nearest-first competitive fill is good enough for thin cuts
        newly = todo & (grown > 0)
        lab[newly] = grown[newly].astype(np.int32)
        todo &= ~newly
    return [lab == i for i in range(1, len(parts) + 1)]


def split_by_line(inst_mask: np.ndarray, path, width: int = 3,
                  min_px: int = 20) -> list:
    """Cut an instance along a user polyline. Returns list of bool masks
    (>=2 when the line actually separates the instance)."""
    cut = paint_stamp(inst_mask.shape, path, max(1, width // 2))
    remain = inst_mask & ~cut
    parts = _components(remain, min_px)
    if len(parts) < 2:
        return [inst_mask]
    return _grow_into(parts, inst_mask & cut)


def split_by_points(inst_mask: np.ndarray, points, min_px: int = 20) -> list:
    """Marker-controlled watershed: one seed click per plant inside a merged
    instance. Pixels flow to the nearest seed following the distance
    transform (so the split follows the natural 'neck' between plants)."""
    pts = [(int(x), int(y)) for x, y in (points or [])]
    pts = [(x, y) for x, y in pts
           if 0 <= y < inst_mask.shape[0] and 0 <= x < inst_mask.shape[1]
           and inst_mask[y, x]]
    if len(pts) < 2:
        return [inst_mask]
    dist = cv2.distanceTransform(inst_mask.astype(np.uint8), cv2.DIST_L2, 5)
    dist = (255 * (1 - dist / max(dist.max(), 1e-6))).astype(np.uint8)
    img3 = cv2.cvtColor(dist, cv2.COLOR_GRAY2BGR)
    markers = np.zeros(inst_mask.shape, np.int32)
    markers[~inst_mask] = 1                   # background marker
    for i, (x, y) in enumerate(pts, 2):
        cv2.circle(markers, (x, y), 3, i, -1)
    markers[~inst_mask] = 1
    cv2.watershed(img3, markers)
    parts = []
    for i in range(2, len(pts) + 2):
        p = (markers == i) & inst_mask
        if p.sum() >= min_px:
            parts.append(p)
    if len(parts) < 2:
        return [inst_mask]
    # watershed ridge lines (-1) -> hand them to the nearest part
    leftover = inst_mask & ~np.any(parts, axis=0)
    return _grow_into(parts, leftover)


def split_by_box(inst_mask: np.ndarray, box, min_px: int = 20) -> list:
    """Everything of the instance inside the box becomes its own instance."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, x1 = sorted((max(0, x0), min(inst_mask.shape[1], x1)))
    y0, y1 = sorted((max(0, y0), min(inst_mask.shape[0], y1)))
    inside = np.zeros(inst_mask.shape, bool)
    inside[y0:y1, x0:x1] = True
    a = inst_mask & inside
    b = inst_mask & ~inside
    if a.sum() < min_px or b.sum() < min_px:
        return [inst_mask]
    return [b, a]


def instance_stats(mask: np.ndarray) -> dict:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {"bbox": None, "area_px": 0, "centroid": None}
    return {"bbox": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
            "area_px": int(len(xs)),
            "centroid": [float(xs.mean()), float(ys.mean())]}
