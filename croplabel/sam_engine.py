"""SAM3 engine: concept-prompt auto-labeling + interactive point/box refinement.

Loads facebook/sam3 lazily (gated model — needs a Hugging Face token with the
license accepted). Every public method degrades gracefully: if SAM3 is not
available, auto-label falls back to classical image processing and point
clicks fall back to a flood-fill magic wand, so the tool always works.
"""
import threading
import traceback

import cv2
import numpy as np


class SamEngine:
    def __init__(self, model_id: str = "facebook/sam3",
                 sam2_id: str = "facebook/sam2.1-hiera-large"):
        self.model_id = model_id
        self.sam2_id = sam2_id
        self.lock = threading.Lock()
        self.model = None
        self.processor = None
        self.tracker = None
        self.tracker_processor = None
        self.backend = None          # 'sam3' | 'sam2' | None
        self.device = None
        self.loading = False
        self.error = None
        self._load_thread = None

    # ------------------------------------------------------------------
    # Loading / auth
    # ------------------------------------------------------------------
    def status(self) -> dict:
        try:
            from huggingface_hub import get_token
            has_token = get_token() is not None
        except Exception:
            has_token = False
        return {
            "loaded": self.model is not None or self.tracker is not None,
            "backend": self.backend,
            "concept_prompts": self.model is not None,
            "tracker_loaded": self.tracker is not None,
            "loading": self.loading,
            "error": self.error,
            "device": self.device,
            "hf_token_present": has_token,
            "model_id": self.model_id,
        }

    def hf_login(self, token: str) -> dict:
        try:
            from huggingface_hub import login, whoami
            login(token=token.strip(), add_to_git_credential=False)
            who = whoami()
            self.error = None
            return {"ok": True, "user": who.get("name", "?")}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def load_async(self):
        with self.lock:
            if self.model is not None or self.loading:
                return
            self.loading = True
            self.error = None
        self._load_thread = threading.Thread(target=self._load, daemon=True)
        self._load_thread.start()

    def ensure_loaded(self, timeout: float = 900):
        self.load_async()
        if self._load_thread is not None:
            self._load_thread.join(timeout=timeout)
        return self.model is not None or self.tracker is not None

    def _load(self):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        try:
            from transformers import Sam3Model, Sam3Processor
            model = Sam3Model.from_pretrained(self.model_id, dtype=dtype)
            model.to(self.device).eval()
            processor = Sam3Processor.from_pretrained(self.model_id)
            self.model, self.processor = model, processor
            self.backend = "sam3"
            # Interactive tracker (SAM2-style point/box prompts) — optional
            try:
                from transformers import Sam3TrackerModel, Sam3TrackerProcessor
                tracker = Sam3TrackerModel.from_pretrained(self.model_id, dtype=dtype)
                tracker.to(self.device).eval()
                self.tracker = tracker
                self.tracker_processor = Sam3TrackerProcessor.from_pretrained(self.model_id)
            except Exception:
                self.tracker = None
                self.tracker_processor = None
        except Exception as e:
            sam3_err = f"{type(e).__name__}: {e}"
            if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
                sam3_err = ("SAM3 access not approved yet on this Hugging Face account "
                            "(huggingface.co/facebook/sam3)")
            traceback.print_exc()
            # Fall back to SAM2.1 — open model, same point/box interface
            try:
                from transformers import Sam2Model, Sam2Processor
                tracker = Sam2Model.from_pretrained(self.sam2_id, dtype=dtype)
                tracker.to(self.device).eval()
                self.tracker = tracker
                self.tracker_processor = Sam2Processor.from_pretrained(self.sam2_id)
                self.backend = "sam2"
                self.error = f"{sam3_err} — using SAM2.1 instead (no text prompts)"
            except Exception as e2:
                self.error = f"{sam3_err}; SAM2 fallback also failed: {e2}"
                traceback.print_exc()
        finally:
            self.loading = False

    # ------------------------------------------------------------------
    # Concept segmentation (text prompt)
    # ------------------------------------------------------------------
    def concept_masks(self, bgr: np.ndarray, prompt: str, threshold: float = 0.4):
        """Return list of {mask(bool HxW), score, box} for a text prompt."""
        if self.model is None:
            return None
        import torch
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=img, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        h, w = bgr.shape[:2]
        post = getattr(self.processor, "post_process_instance_segmentation", None)
        results = post(outputs, threshold=threshold, mask_threshold=0.5,
                       target_sizes=[(h, w)])[0]
        out = []
        masks = results.get("masks")
        scores = results.get("scores")
        boxes = results.get("boxes")
        if masks is None:
            return out
        for i in range(len(masks)):
            m = masks[i]
            m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
            sc = float(scores[i]) if scores is not None else 1.0
            bx = [float(v) for v in boxes[i]] if boxes is not None else None
            out.append({"mask": m.astype(bool), "score": sc, "box": bx})
        return out

    # ------------------------------------------------------------------
    # Interactive point / box segmentation
    # ------------------------------------------------------------------
    def point_mask(self, bgr: np.ndarray, points, labels, prefer: str = "iou"):
        """points: [[x,y],...] labels: [1|0,...] -> bool mask or None.

        prefer='iou' returns SAM's highest-confidence mask (often the WHOLE
        object — on a FAF scan that is the entire retina disc).
        prefer='small' returns the smallest near-best mask instead — what a
        click on a single lesion should give."""
        if self.tracker is None:
            return None
        try:
            import torch
            from PIL import Image
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            inputs = self.tracker_processor(
                images=img,
                input_points=[[[list(map(float, p)) for p in points]]],
                input_labels=[[[int(l) for l in labels]]],
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.tracker(**inputs, multimask_output=True)
            masks = self.tracker_processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
            masks = np.asarray(masks).reshape(-1, bgr.shape[0], bgr.shape[1])
            iou = outputs.iou_scores.detach().float().cpu().numpy().reshape(-1)
            n = len(masks)
            if prefer == "small":
                areas = masks.reshape(n, -1).sum(1)
                best_iou = float(iou[:n].max())
                cand = [i for i in range(n)
                        if areas[i] > 0 and iou[i] >= best_iou - 0.15]
                best = (min(cand, key=lambda i: areas[i]) if cand
                        else int(np.argmax(iou[:n])))
            else:
                best = int(np.argmax(iou[:n]))
            return masks[best].astype(bool)
        except Exception:
            traceback.print_exc()
            return None

    def box_mask(self, bgr: np.ndarray, box):
        """box: [x0,y0,x1,y1] -> bool mask or None."""
        if self.tracker is None:
            return None
        try:
            import torch
            from PIL import Image
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            inputs = self.tracker_processor(
                images=img, input_boxes=[[list(map(float, box))]],
                return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.tracker(**inputs, multimask_output=False)
            masks = self.tracker_processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
            masks = np.asarray(masks).reshape(-1, bgr.shape[0], bgr.shape[1])
            return masks[0].astype(bool)
        except Exception:
            traceback.print_exc()
            return None


# ---------------------------------------------------------------------------
# SAM3.1 native (official sam3 package + local checkpoint, GPU) — the primary
# engine: full-frame text-prompt concept segmentation with adjustable
# threshold. ~0.3s for image features + all prompts on this machine's GPU.
# ---------------------------------------------------------------------------
class Sam3NativeEngine:
    def __init__(self, ckpt_path: str, threshold: float = 0.4):
        from pathlib import Path
        self.ckpt = Path(ckpt_path)
        self.threshold = threshold
        self.lock = threading.Lock()
        self.model = None
        self.proc = None
        self.error = None

    def available(self) -> bool:
        try:
            import torch
            return self.ckpt.exists() and torch.cuda.is_available()
        except Exception:
            return False

    def load(self) -> bool:
        with self.lock:
            if self.proc is not None:
                return True
            if not self.available():
                self.error = f"SAM3.1 checkpoint or GPU not available ({self.ckpt})"
                return False
            try:
                import torch
                from sam3 import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor as NativeProc
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                self.model = build_sam3_image_model(
                    device="cuda", load_from_HF=False, checkpoint_path=str(self.ckpt))
                self.proc = NativeProc(self.model, device="cuda",
                                       confidence_threshold=self.threshold)
                return True
            except Exception as e:
                self.error = f"SAM3.1 native load failed: {e}"
                traceback.print_exc()
                self.model = self.proc = None
                return False

    def concept_instances(self, bgr: np.ndarray, prompts, threshold=None,
                          enhance: bool = True):
        """Run text prompts full-frame (raw + CLAHE variant). Returns
        {prompt: [ {mask, score, box}, ... ]} merged over variants, or None."""
        if not self.load():
            return None
        import torch
        from PIL import Image
        self.proc.confidence_threshold = float(threshold or self.threshold)
        variants = [bgr]
        if enhance:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16)).apply(gray)
            variants.append(cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR))
        out = {p: [] for p in prompts}
        with self.lock, torch.autocast("cuda", dtype=torch.bfloat16), \
                torch.inference_mode():
            for img_bgr in variants:
                img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                state = self.proc.set_image(img)
                for p in prompts:
                    self.proc.reset_all_prompts(state)
                    state = self.proc.set_text_prompt(p, state)
                    n = len(state["scores"])
                    for i in range(n):
                        out[p].append({
                            "mask": state["masks"][i, 0].cpu().numpy().astype(bool),
                            "score": float(state["scores"][i]),
                            "box": [float(v) for v in state["boxes"][i]],
                        })
        return out

    def exemplar_instances(self, bgr: np.ndarray, pos_boxes, neg_boxes=None,
                           text: str = None, threshold=None):
        """Visual-exemplar segmentation: the user boxes one or two example
        lesions (and optionally counter-examples, e.g. eyelash shadows) and
        SAM3 finds every similar-looking instance across the whole frame.
        Boxes are pixel [x0,y0,x1,y1]. Returns [{mask, score, box}] or None.

        Validated on this dataset: ONE positive exemplar on a labeled spot
        reproduced the full manual disease mask at 0.90 precision /
        0.96 recall (84 instances found)."""
        if not self.load():
            return None
        import torch
        from PIL import Image
        h, w = bgr.shape[:2]

        def _norm(b):  # xyxy px -> normalized cxcywh
            x0, y0, x1, y1 = (float(v) for v in b)
            return [((x0 + x1) / 2) / w, ((y0 + y1) / 2) / h,
                    max(x1 - x0, 1) / w, max(y1 - y0, 1) / h]

        out = []
        with self.lock, torch.autocast("cuda", dtype=torch.bfloat16), \
                torch.inference_mode():
            self.proc.confidence_threshold = float(threshold or self.threshold)
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            state = self.proc.set_image(img)
            if text:
                state = self.proc.set_text_prompt(text, state)
            for b in pos_boxes or []:
                state = self.proc.add_geometric_prompt(_norm(b), True, state)
            for b in neg_boxes or []:
                state = self.proc.add_geometric_prompt(_norm(b), False, state)
            for i in range(len(state["scores"])):
                out.append({
                    "mask": state["masks"][i, 0].cpu().numpy().astype(bool),
                    "score": float(state["scores"][i]),
                    "box": [float(v) for v in state["boxes"][i]],
                })
        return out


# ---------------------------------------------------------------------------
# SAM3 via local ONNX export (no gated download needed)
# ---------------------------------------------------------------------------
class Sam3OnnxEngine:
    """Concept (text-prompt) segmentation with the locally exported SAM3 ONNX
    models (image encoder + language encoder + decoder, input 1008).

    The exported decoder has a baked-in presence threshold, and on full-frame
    retinal images lesions are too small to fire — so concept segmentation is
    run on overlapping zoomed tiles and the instance masks are merged.
    Validated empirically: 'spot' on 512px FAF tiles segments the dark
    hypo-autofluorescent lesions cleanly.
    """

    def __init__(self, model_dir: str):
        from pathlib import Path
        self.dir = Path(model_dir)
        self.lock = threading.Lock()
        self.enc = self.lang = self.dec = None
        self.tok = None
        self.error = None
        self._text_cache = {}

    def available(self) -> bool:
        return (self.dir / "sam3_image_encoder.onnx").exists() and \
               (self.dir / "sam3_decoder.onnx").exists() and \
               (self.dir / "sam3_language_encoder.onnx").exists()

    def load(self) -> bool:
        with self.lock:
            if self.dec is not None:
                return True
            if not self.available():
                self.error = f"SAM3 ONNX models not found in {self.dir}"
                return False
            try:
                import onnxruntime as ort
                prov = ["CPUExecutionProvider"]
                self.enc = ort.InferenceSession(str(self.dir / "sam3_image_encoder.onnx"),
                                                providers=prov)
                self.lang = ort.InferenceSession(str(self.dir / "sam3_language_encoder.onnx"),
                                                 providers=prov)
                self.dec = ort.InferenceSession(str(self.dir / "sam3_decoder.onnx"),
                                                providers=prov)
                self._load_tokenizer()
                return True
            except Exception as e:
                self.error = f"SAM3 ONNX load failed: {e}"
                traceback.print_exc()
                self.enc = self.lang = self.dec = None
                return False

    def _load_tokenizer(self):
        """CLIP BPE tokenizer, self-contained: bundled tokenizer.json first
        (works in the exe and offline), transformers as fallback. Uses the
        bare `tokenizers` library so no transformers dependency checks
        (accelerate etc.) can break it inside a PyInstaller bundle."""
        from pathlib import Path
        local = Path(__file__).parent / "assets" / "clip_tokenizer" / "tokenizer.json"
        try:
            from tokenizers import Tokenizer
            if local.exists():
                self.tok = Tokenizer.from_file(str(local))
                return
        except Exception:
            pass
        from transformers import CLIPTokenizerFast
        hf_tok = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
        self.tok = hf_tok.backend_tokenizer

    def _text_feats(self, prompt: str):
        if prompt not in self._text_cache:
            ids = self.tok.encode(prompt).ids[:32]
            tokens = np.zeros((1, 32), np.int64)
            tokens[0, :len(ids)] = ids
            out = self.lang.run(None, {"tokens": tokens})
            names = [o.name for o in self.lang.get_outputs()]
            L = dict(zip(names, out))
            self._text_cache[prompt] = (L["text_attention_mask"], L["text_memory"])
        return self._text_cache[prompt]

    def _encode(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        interp = cv2.INTER_AREA if max(bgr.shape[:2]) > 1008 else cv2.INTER_LANCZOS4
        chw = cv2.resize(rgb, (1008, 1008), interpolation=interp
                         ).transpose(2, 0, 1).astype(np.uint8)
        out = self.enc.run(None, {"image": chw})
        return dict(zip([o.name for o in self.enc.get_outputs()], out))

    def _decode(self, feats, prompt, h, w):
        lm, lf = self._text_feats(prompt)
        boxes, scores, masks = self.dec.run(None, {
            "original_height": np.array(h, np.int64),
            "original_width": np.array(w, np.int64),
            "vision_pos_enc_2": feats["vision_pos_enc_2"],
            "backbone_fpn_0": feats["backbone_fpn_0"],
            "backbone_fpn_1": feats["backbone_fpn_1"],
            "backbone_fpn_2": feats["backbone_fpn_2"],
            "language_mask": lm, "language_features": lf,
            "box_coords": np.zeros((1, 1, 4), np.float32),
            "box_labels": np.zeros((1, 1), np.int64),
            "box_masks": np.array([[False]])})
        return boxes, scores, masks.reshape(-1, h, w) if masks.size else masks.reshape(0, h, w)

    def concept_full(self, bgr: np.ndarray, prompt: str):
        """Single full-frame pass. Returns bool union mask (may be empty)."""
        if not self.load():
            return None
        h, w = bgr.shape[:2]
        _, scores, masks = self._decode(self._encode(bgr), prompt, h, w)
        out = np.zeros((h, w), bool)
        for m in masks:
            out |= m
        return out

    def concept_tiled(self, bgr: np.ndarray, prompts, tile: int = 512,
                      stride: int = 256, progress_cb=None, scales=None,
                      enhance: bool = True):
        """Tiled concept segmentation: union of instance masks over all tiles,
        prompts and scales, mapped back to full-image coordinates. Instances
        clipped by an interior tile border are dropped — the overlapping
        neighbour tile sees them whole, so no straight-edge artifacts.

        With enhance=True every tile is also run through a CLAHE
        contrast-enhanced copy of the image: on dark/noisy FAF scans the raw
        pixels often fire no concept at all while the enhanced ones do
        (verified on patient 81597 R: raw 0 detections, CLAHE 27)."""
        if not self.load():
            return None
        h, w = bgr.shape[:2]
        variants = [bgr]
        if enhance:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16)).apply(gray)
            variants.append(cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR))
        union = np.zeros((h, w), bool)
        if scales is None:
            scales = [(tile, stride), (min(tile * 3 // 2, min(h, w)), stride * 3 // 2)]
        tiles = []
        for t, s in scales:
            xs = sorted({min(x, max(0, w - t)) for x in range(0, w, s) if x < w})
            ys = sorted({min(y, max(0, h - t)) for y in range(0, h, s) if y < h})
            tiles += [(x, y, t) for y in ys for x in xs]
        for i, (x, y, t) in enumerate(tiles):
            for img in variants:
                sub = img[y:y + t, x:x + t]
                th, tw = sub.shape[:2]
                feats = self._encode(sub)
                for p in prompts:
                    _, scores, masks = self._decode(feats, p, th, tw)
                    for m in masks:
                        if self._clipped_by_tile(m, x, y, th, tw, h, w):
                            continue
                        union[y:y + th, x:x + tw] |= m
            if progress_cb:
                progress_cb(i + 1, len(tiles))
        return union

    @staticmethod
    def _clipped_by_tile(m, x, y, th, tw, img_h, img_w):
        """True if the instance touches a tile border that is NOT an image border."""
        if y > 0 and m[0, :].any():
            return True
        if y + th < img_h and m[-1, :].any():
            return True
        if x > 0 and m[:, 0].any():
            return True
        if x + tw < img_w and m[:, -1].any():
            return True
        return False


# ---------------------------------------------------------------------------
# Classical fallbacks (work with no model / no internet)
# ---------------------------------------------------------------------------
def magic_wand(bgr: np.ndarray, x: int, y: int, tolerance: int = 18) -> np.ndarray:
    """Flood-fill region grow from a click — fallback for smart click."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    cv2.floodFill(gray, mask, (int(x), int(y)), 0,
                  loDiff=tolerance, upDiff=tolerance, flags=flags)
    return mask[1:-1, 1:-1] > 0


def detect_optic_disc(bgr: np.ndarray, content: np.ndarray):
    """Rough optic disc detection on FAF: small dark round blob near centre.
    Returns (mask, (cx, cy)) or (None, None)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blur = cv2.medianBlur(gray, 9)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=2, minDist=200,
                               param1=80, param2=30,
                               minRadius=h // 80, maxRadius=h // 18)
    if circles is None:
        return None, None
    cx0, cy0 = w / 2, h / 2
    best, best_d = None, 1e18
    for c in circles[0]:
        x, y, r = c
        if not (0 <= int(y) < h and 0 <= int(x) < w) or not content[int(y), int(x)]:
            continue
        d = (x - cx0) ** 2 + (y - cy0) ** 2
        if d < best_d:
            best, best_d = c, d
    if best is None:
        return None, None
    x, y, r = best
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (int(x), int(y)), int(r), 1, -1)
    return mask.astype(bool), (float(x), float(y))


def detect_dark_lesions(bgr: np.ndarray, content: np.ndarray,
                        rel_thresh: float = 0.45, min_area: int = 25):
    """Fallback disease detection on FAF: hypo-autofluorescent (dark) patches
    well inside the retina content."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    inside = cv2.erode(content.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    vals = gray[inside]
    if vals.size == 0:
        return np.zeros(gray.shape, bool)
    ref = np.percentile(vals, 70)
    m = ((gray < ref * rel_thresh) & inside).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(gray.shape, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep |= lab == i
    return keep
