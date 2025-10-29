from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple
from collections import OrderedDict

import numpy as np
from PyQt6.QtGui import QImage, QImageReader

try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]


FrameLoader = Callable[[int], Optional[np.ndarray]]


@dataclass
class FrameSource:
    total_frames: int
    fps: float
    loader: FrameLoader
    is_video: bool = False
    is_animated: bool = False


@dataclass
class PreviewState:
    current_frame: int = 0
    hsv: Dict[str, float] = field(default_factory=lambda: {
        "tol_h": 10,
        "tol_s": 60,
        "tol_v": 60,
        "strength": 1.5,
        "erode_iter": 1,
        "dilate_iter": 0,
        "feather_px": 2.0,
    })
    seed: Optional[Tuple[int, int]] = None


@dataclass
class PreviewFrame:
    image: QImage
    overlay: Optional[QImage]
    meta: Dict[str, object]


class PreviewService:
    def __init__(self) -> None:
        self._source: Optional[FrameSource] = None
        self._state = PreviewState()
        self._cache: "OrderedDict[Tuple[str, int], np.ndarray]" = OrderedDict()
        self._cache_cap = 3

    # --- Source preparation ---
    def set_static_source(self, path: str) -> None:
        cv2_mod = self._require_cv2()

        def loader(_: int) -> Optional[np.ndarray]:
            frame = cv2_mod.imread(path, cv2_mod.IMREAD_COLOR)
            if frame is None:
                return None
            return self._normalize_frame(frame)

        self._source = FrameSource(total_frames=1, fps=0.0, loader=loader, is_video=False, is_animated=False)
        self._state.current_frame = 0
        self._cache.clear()

    def set_video_source(self, path: str) -> None:
        cv2_mod = self._require_cv2()
        cap = cv2_mod.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("無法開啟影片來源")
        total = int(cap.get(cv2_mod.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(cap.get(cv2_mod.CAP_PROP_FPS)) if cap.get(cv2_mod.CAP_PROP_FPS) else 0.0
        cap.release()
        if total <= 0:
            total = 1

        def loader(index: int) -> Optional[np.ndarray]:
            capture = cv2_mod.VideoCapture(path)
            if not capture.isOpened():
                return None
            try:
                capture.set(cv2_mod.CAP_PROP_POS_FRAMES, float(index))
                ok, frame = capture.read()
                if not ok or frame is None:
                    return None
                return self._normalize_frame(frame)
            finally:
                capture.release()

        self._source = FrameSource(total_frames=total, fps=fps, loader=loader, is_video=True, is_animated=False)
        self._state.current_frame = 0
        self._cache.clear()

    def set_animated_source(self, path: str) -> None:
        reader = QImageReader(path)
        reader.setDecideFormatFromContent(True)
        image_count = reader.imageCount()
        total = int(image_count) if image_count and image_count > 0 else 1

        def loader(index: int) -> Optional[np.ndarray]:
            local_reader = QImageReader(path)
            local_reader.setDecideFormatFromContent(True)
            try:
                if hasattr(local_reader, 'jumpToImage'):
                    local_reader.jumpToImage(int(index))
                elif hasattr(local_reader, 'setCurrentImageNumber'):
                    local_reader.setCurrentImageNumber(int(index))
            except Exception:
                pass
            image = local_reader.read()
            if image is None or image.isNull():
                return None
            image = image.convertToFormat(QImage.Format.Format_RGB888)
            width = image.width()
            height = image.height()
            ptr = image.bits()
            ptr.setsize(height * width * 3)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
            return self._normalize_frame(arr[:, :, ::-1].copy())

        self._source = FrameSource(total_frames=total, fps=0.0, loader=loader, is_video=False, is_animated=True)
        self._state.current_frame = 0
        self._cache.clear()

    def clear_cache(self) -> None:
        self._cache.clear()

    # --- State helpers ---
    def get_source_info(self) -> Dict[str, object]:
        if not self._source:
            return {"total_frames": 1, "fps": 0.0, "is_video": False, "is_animated": False}
        return {
            "total_frames": self._source.total_frames,
            "fps": self._source.fps,
            "is_video": self._source.is_video,
            "is_animated": self._source.is_animated,
        }

    def get_state(self) -> PreviewState:
        return self._state

    def update_hsv(self, key: str, value: float) -> None:
        self._state.hsv[key] = value

    def update_seed(self, seed: Optional[Tuple[int, int]]) -> None:
        self._state.seed = seed

    def seek(self, index: int) -> None:
        if not self._source:
            return
        index = max(0, min(int(index), max(0, self._source.total_frames - 1)))
        self._state.current_frame = index

    # --- Frame retrieval ---
    def load_current_frame(self, *, apply_hsv: bool = True) -> PreviewFrame:
        if not self._source:
            raise RuntimeError("Preview source not prepared")
        frame = self._get_frame(self._state.current_frame)
        if frame is None:
            raise RuntimeError("讀取影格失敗")
        overlay = self._compute_overlay(frame) if apply_hsv else None
        return self._to_preview_frame(frame, overlay)

    # --- Internal helpers ---
    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        elif frame.shape[2] == 4:
            frame = frame[:, :, :3]
        return np.ascontiguousarray(frame)

    def _get_frame(self, index: int) -> Optional[np.ndarray]:
        key = ("frame", int(index))
        if key in self._cache:
            cached = self._cache.pop(key)
            self._cache[key] = cached
            return cached.copy()
        if not self._source:
            return None
        frame = self._source.loader(index)
        if frame is not None:
            frame = frame.copy()
            self._cache[key] = frame
            while len(self._cache) > self._cache_cap:
                self._cache.popitem(last=False)
        return frame

    def _compute_overlay(self, bgr: np.ndarray) -> Optional[QImage]:
        try:
            from core.hsv_bg import compute_alpha
        except Exception:
            return None
        opts = {
            "tol_h": int(self._state.hsv["tol_h"]),
            "tol_s": int(self._state.hsv["tol_s"]),
            "tol_v": int(self._state.hsv["tol_v"]),
            "strength": float(self._state.hsv["strength"]),
            "erode_iter": int(self._state.hsv["erode_iter"]),
            "dilate_iter": int(self._state.hsv["dilate_iter"]),
            "feather_px": float(self._state.hsv["feather_px"]),
        }
        try:
            alpha = compute_alpha(bgr.copy(), opts)
        except Exception:
            return None
        h, w = alpha.shape[:2]
        overlay_arr = np.zeros((h, w, 4), dtype=np.uint8)
        overlay_arr[..., 1] = 255
        overlay_arr[..., 3] = (alpha > 0).astype(np.uint8) * 90
        overlay_arr = np.ascontiguousarray(overlay_arr)
        return QImage(
            overlay_arr.tobytes(),
            w,
            h,
            overlay_arr.strides[0],
            QImage.Format.Format_RGBA8888,
        ).copy()

    def _to_preview_frame(self, bgr: np.ndarray, overlay: Optional[QImage]) -> PreviewFrame:
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        h, w = rgb.shape[:2]
        qimg = QImage(
            rgb.tobytes(),
            w,
            h,
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        meta = {
            "frame": self._state.current_frame,
            "fps": self._source.fps if self._source else 0.0,
            "total_frames": self._source.total_frames if self._source else 1,
            "bgr": bgr.copy(),
        }
        return PreviewFrame(image=qimg, overlay=overlay, meta=meta)

    def _require_cv2(self) -> Any:
        if cv2 is None:
            raise RuntimeError("需要安裝 opencv-python 才能使用預覽功能。")
        return cv2
