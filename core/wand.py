from __future__ import annotations
import numpy as np
import cv2
from typing import Tuple, Dict, Any


def compute_mask(bgr: np.ndarray, seed_xy: Tuple[int, int], opts: Dict[str, Any]) -> np.ndarray:
    """Compute magic-wand foreground mask (255=FG) from BGR image.

    Options (with defaults):
      - tolH/tolS/tolV: int (Hue/Sat/Val tolerance)
      - contiguous: bool (floodFill) or global inRange when False
      - connectivity: 4|8
      - fixed_range: bool (floodFill flag)
      - use_edge_barrier: bool (add Canny barrier to floodFill mask)
      - near_radius_ratio: float (merge nearby components around main center)
      - min_keep_area: int (min area to consider merging)
      - morph_open_close: bool
      - feather_px: float (distance transform feathering radius)
      - invert: bool (invert final mask)
    """
    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")

    o = {
        "tolH": 10, "tolS": 60, "tolV": 60,
        "contiguous": True, "connectivity": 8,
        "fixed_range": False, "use_edge_barrier": True,
        "near_radius_ratio": 0.35, "min_keep_area": 400,
        "morph_open_close": True, "feather_px": 2.0,
        "invert": False,
    }
    if opts:
        o.update(opts)

    h, w = bgr.shape[:2]
    x = int(max(0, min(w - 1, seed_xy[0])))
    y = int(max(0, min(h - 1, seed_xy[1])))

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    if not o["contiguous"]:
        Hs, Ss, Vs = map(int, hsv[y, x])
        lo = (max(0, Hs - int(o["tolH"])), max(0, Ss - int(o["tolS"])) , max(0, Vs - int(o["tolV"])) )
        hi = (min(179, Hs + int(o["tolH"])), min(255, Ss + int(o["tolS"])) , min(255, Vs + int(o["tolV"])) )
        mask = cv2.inRange(hsv, lo, hi)
    else:
        ff_mask = np.zeros((h + 2, w + 2), np.uint8)
        if o["use_edge_barrier"]:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            med = float(np.median(gray))
            high = int(np.clip(1.33 * med, 60, 220))
            low = max(1, high // 3)
            edges = cv2.Canny(gray, low, high)
            bar = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
            bar = cv2.dilate(bar, np.ones((3, 3), np.uint8), 1)
            ff_mask[1:-1, 1:-1][bar > 0] = 255
        flags = cv2.FLOODFILL_MASK_ONLY | (4 if int(o["connectivity"]) == 4 else 8)
        if o["fixed_range"]:
            flags |= cv2.FLOODFILL_FIXED_RANGE
        loDiff = (int(o["tolH"]), int(o["tolS"]), int(o["tolV"]))
        upDiff = (int(o["tolH"]), int(o["tolS"]), int(o["tolV"]))
        cv2.floodFill(hsv.copy(), ff_mask, (x, y), (0, 0, 0), loDiff, upDiff, flags)
        mask = (ff_mask[1:-1, 1:-1] > 0).astype(np.uint8) * 255

    if o.get("invert", False):
        mask = 255 - mask

    # 連通域合併與去雜
    num, labels, stats, cents = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num > 1:
        areas = stats[:, cv2.CC_STAT_AREA]
        areas[0] = 0
        main = int(np.argmax(areas))
        keep = (labels == main).astype(np.uint8) * 255
        cx, cy = cents[main]
        R = float(o["near_radius_ratio"]) * max(h, w)
        for i in range(1, num):
            if i == main:
                continue
            if stats[i, cv2.CC_STAT_AREA] < int(o["min_keep_area"]):
                continue
            if np.hypot(cents[i][0] - cx, cents[i][1] - cy) <= R:
                keep |= (labels == i).astype(np.uint8) * 255
        mask = keep

    if o["morph_open_close"]:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    if float(o["feather_px"]) > 0:
        dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
        alpha = np.clip(dist / float(o["feather_px"]), 0.0, 1.0)
        mask = (alpha * 255).astype(np.uint8)

    return mask

