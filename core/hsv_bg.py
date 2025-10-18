from __future__ import annotations
import numpy as np
import cv2
from typing import Dict, Any


def compute_alpha(bgr: np.ndarray, opts: Dict[str, Any]) -> np.ndarray:
    """HSV-based background removal alpha (0-255) from BGR image.

    Options keys (with defaults):
      tol_h=10, tol_s=60, tol_v=60, strength=1.5,
      erode_iter=1, dilate_iter=0, feather_px=2.0, use_guided=False
    """
    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")

    o = {
        "tol_h": 10, "tol_s": 60, "tol_v": 60,
        "strength": 1.5,
        "erode_iter": 1, "dilate_iter": 0,
        "feather_px": 2.0,
        "use_guided": False,
    }
    if opts:
        o.update(opts)

    # ensure 3-channel BGR
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    elif bgr.shape[2] == 4:
        bgr = bgr[:, :, :3]

    # HSV inRange region from dynamic background sampling (borders)
    # We reuse the algorithmic spirit from threads: estimate background HSV by borders,
    # then threshold and generate soft alpha by distance transform.

    # Estimate background HSV range by borders using median + tolerance
    H, W = bgr.shape[:2]
    pad = max(1, min(12, H // 4 if H > 1 else 1, W // 4 if W > 1 else 1))
    border = np.concatenate([
        bgr[:pad, :, :].reshape(-1, 3),
        bgr[-pad:, :, :].reshape(-1, 3),
        bgr[:, :pad, :].reshape(-1, 3),
        bgr[:, -pad:, :].reshape(-1, 3)
    ], axis=0)
    if border.size == 0:
        border = bgr.reshape(-1, 3)
    hsv_border = cv2.cvtColor(border.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hb = int(np.median(hsv_border[:, 0]))
    sb = int(np.median(hsv_border[:, 1]))
    vb = int(np.median(hsv_border[:, 2]))
    k = float(o["strength"])
    th, ts, tv = int(round(o["tol_h"] * k)), int(round(o["tol_s"] * k)), int(round(o["tol_v"] * k))

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo = (max(0, hb - th), max(0, sb - ts), max(0, vb - tv))
    hi = (min(179, hb + th), min(255, sb + ts), min(255, vb + tv))
    bg = cv2.inRange(hsv, lo, hi)  # background=255

    if int(o["dilate_iter"]) > 0:
        k3 = np.ones((3, 3), np.uint8)
        bg = cv2.dilate(bg, k3, iterations=int(o["dilate_iter"]))
    if int(o["erode_iter"]) > 0:
        k3 = np.ones((3, 3), np.uint8)
        bg = cv2.erode(bg, k3, iterations=int(o["erode_iter"]))

    fg_bin = cv2.bitwise_not(bg)  # 255=FG
    if float(o["feather_px"]) > 0:
        dist = cv2.distanceTransform(fg_bin, cv2.DIST_L2, 3)
        soft = np.clip(dist / float(o["feather_px"]), 0.0, 1.0)
        alpha = (soft * 255.0).astype(np.uint8)
    else:
        alpha = fg_bin

    if bool(o.get("use_guided", False)):
        try:
            import cv2.ximgproc as xip
            guide = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            alpha = xip.guidedFilter(guide, alpha, radius=4, eps=1e-3)
        except Exception:
            pass

    return alpha
