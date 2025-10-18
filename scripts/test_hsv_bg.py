#!/usr/bin/env python3
from __future__ import annotations
import os
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

# ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.hsv_bg import compute_alpha


def synthetic_sample(w=640, h=360):
    bg = np.zeros((h, w, 3), np.uint8)
    # green-ish background gradient
    for y in range(h):
        g = int(120 + 60 * (y / max(1, h-1)))
        bg[y, :, :] = (40, g, 40)
    # red circle as foreground
    cv2.circle(bg, (w//2, h//2), min(w,h)//6, (40,40,220), thickness=-1)
    # blue rectangle
    cv2.rectangle(bg, (w//6, h//3), (w//3, h//2), (220,40,40), thickness=-1)
    return bg


def overlay_preview(bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    ov = bgr.copy()
    mask = (alpha > 0).astype(np.uint8)
    # green overlay on FG
    green = np.zeros_like(bgr)
    green[:, :, 1] = 255
    mix = (0.65 * ov + 0.35 * green).astype(np.uint8)
    ov[mask > 0] = mix[mask > 0]
    return ov


def main():
    ap = argparse.ArgumentParser(description="Test core.hsv_bg.compute_alpha")
    ap.add_argument('--input', '-i', help='Input image path. If omitted, use synthetic sample.')
    ap.add_argument('--outdir', '-o', default='./_tmp_hsv_test', help='Output directory')
    ap.add_argument('--tol_h', type=int, default=10)
    ap.add_argument('--tol_s', type=int, default=60)
    ap.add_argument('--tol_v', type=int, default=60)
    ap.add_argument('--strength', type=float, default=1.5)
    ap.add_argument('--erode', type=int, default=1)
    ap.add_argument('--dilate', type=int, default=0)
    ap.add_argument('--feather', type=float, default=2.0)
    ap.add_argument('--guided', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.input:
        bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"Failed to read image: {args.input}")
        name = os.path.splitext(os.path.basename(args.input))[0]
    else:
        bgr = synthetic_sample()
        name = 'synthetic'

    opts = {
        'tol_h': args.tol_h,
        'tol_s': args.tol_s,
        'tol_v': args.tol_v,
        'strength': args.strength,
        'erode_iter': args.erode,
        'dilate_iter': args.dilate,
        'feather_px': args.feather,
        'use_guided': args.guided,
    }
    alpha = compute_alpha(bgr, opts)
    fg_ratio = float((alpha > 0).sum()) / float(alpha.size)
    print(f"alpha shape={alpha.shape}, dtype={alpha.dtype}, fg_ratio={fg_ratio:.3f}")

    ov = overlay_preview(bgr, alpha)
    cv2.imwrite(os.path.join(args.outdir, f"{name}_input.png"), bgr)
    cv2.imwrite(os.path.join(args.outdir, f"{name}_alpha.png"), alpha)
    cv2.imwrite(os.path.join(args.outdir, f"{name}_overlay.png"), ov)
    print(f"Saved results into {args.outdir}")


if __name__ == '__main__':
    main()
