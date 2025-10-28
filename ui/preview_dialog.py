from __future__ import annotations
from typing import Optional, Callable
from PyQt6.QtCore import Qt, pyqtSignal, QEvent
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QSlider
from PyQt6.QtGui import QPixmap, QImage, QPainter, QImageReader
import numpy as np
import cv2
import os
from collections import OrderedDict


class PreviewImageDialog(QDialog):
    """影片模式單影格預覽 + HSV 預覽 + 魔術棒取樣。

    - 顯示目前影格（圖片）。
    - 可切換/調整 HSV 參數並覆蓋預覽遮色。
    - 可點擊取樣（記錄 wand seed）。
    - 可重新擷取影格（由外部 refresh_fn 提供）。

    訊號：
    - hsvChanged(dict)
    - seedSelected(tuple[int,int])
    """

    hsvChanged = pyqtSignal(dict)
    seedSelected = pyqtSignal(object)

    def __init__(self, img_path: str, refresh_fn: Optional[Callable[[], Optional[str]]] = None,
                 init_hsv: Optional[dict] = None, init_ms: Optional[int] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("影格預覽")
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("QWidget { background:#222; color:#fff; font-family:'Source Han Sans TC'; }")

        self._img_path = img_path
        self._src_path = img_path  # 兼容命名，新增影片/動圖模式
        self._refresh_fn = refresh_fn
        self._seed: Optional[tuple[int,int]] = None
        self._hsv = {
            "tol_h": 10, "tol_s": 60, "tol_v": 60,
            "strength": 1.5,
            "erode_iter": 1, "dilate_iter": 0,
            "feather_px": 2.0,
        }
        if isinstance(init_hsv, dict):
            self._hsv.update({k: init_hsv.get(k, self._hsv[k]) for k in self._hsv.keys()})

        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(8)

        self.info = QLabel("點擊圖片以取樣（魔術棒 seed）/ 調整 HSV 預覽")
        v.addWidget(self.info)

        self.view = QLabel()
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumSize(640, 360)
        self.view.setStyleSheet("QLabel { background:#111; border:1px solid #333; border-radius:8px; }")
        v.addWidget(self.view, 1)
        self.view.installEventFilter(self)

        # 幀選擇 UI（影片/動圖模式下顯示）
        row_f = QHBoxLayout()
        self.frame_label = QLabel("Frame: -/-  (00:00)")
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        row_f.addWidget(self.frame_label)
        row_f.addWidget(self.frame_slider)
        v.addLayout(row_f)

        # HSV 控制
        row1 = QHBoxLayout()
        self.sl_h = QSlider(Qt.Orientation.Horizontal); self.sl_h.setRange(1, 60); self.sl_h.setValue(int(self._hsv["tol_h"]))
        self.sl_s = QSlider(Qt.Orientation.Horizontal); self.sl_s.setRange(1, 100); self.sl_s.setValue(int(self._hsv["tol_s"]))
        self.sl_v = QSlider(Qt.Orientation.Horizontal); self.sl_v.setRange(1, 100); self.sl_v.setValue(int(self._hsv["tol_v"]))
        row1.addWidget(QLabel("H")); row1.addWidget(self.sl_h)
        row1.addWidget(QLabel("S")); row1.addWidget(self.sl_s)
        row1.addWidget(QLabel("V")); row1.addWidget(self.sl_v)
        v.addLayout(row1)

        row2 = QHBoxLayout()
        self.sl_strength = QSlider(Qt.Orientation.Horizontal); self.sl_strength.setRange(50, 300); self.sl_strength.setValue(int(float(self._hsv["strength"]) * 100))
        row2.addWidget(QLabel("倍率")); row2.addWidget(self.sl_strength)
        v.addLayout(row2)

        # 操作按鈕
        btns = QHBoxLayout(); btns.addStretch(1)
        self.btn_reload = QPushButton("重新擷取")
        self.btn_apply_hsv = QPushButton("套用到控制")
        self.btn_use_seed = QPushButton("使用取樣座標")
        self.btn_close = QPushButton("關閉")
        for b in (self.btn_reload, self.btn_apply_hsv, self.btn_use_seed, self.btn_close):
            b.setStyleSheet("QPushButton { background:#333; color:#EEE; border:none; border-radius:6px; padding:6px 12px; }")
            btns.addWidget(b)
        v.addLayout(btns)

        self.btn_reload.clicked.connect(self._on_reload)
        self.btn_apply_hsv.clicked.connect(self._on_apply_hsv)
        self.btn_use_seed.clicked.connect(self._on_use_seed)
        self.btn_close.clicked.connect(self.accept)

        for s in (self.sl_h, self.sl_s, self.sl_v, self.sl_strength):
            s.valueChanged.connect(self._update_hsv_preview)

        # 媒體屬性
        self._is_video = False
        self._is_animated_image = False
        self._total_frames = 1
        self._fps = 0.0
        self._current_frame = 0
        self._init_ms = int(init_ms) if isinstance(init_ms, (int, float)) else None
        self._reader = None
        self._cap = None
        # LRU frame cache (keep last 3 frames)
        self._frame_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._cache_cap = 3

        # 連動
        self.frame_slider.valueChanged.connect(self._on_seek_frame)

        # 初始化媒體並載入
        self._init_media()
        self._load_current()

    def _load(self, path: str):
        self._bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if self._bgr is None:
            self.view.setText("讀取圖片失敗")
            return
        rgb = cv2.cvtColor(self._bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self._qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._set_display(self._qimg)
        self._update_hsv_preview()

    def _set_display(self, qimg: QImage, overlay: Optional[QImage] = None):
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.view.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        if overlay is not None:
            ov = QImage(overlay)
            ov = ov.scaled(scaled.size(), Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
            out = QPixmap(scaled.size()); out.fill(Qt.GlobalColor.transparent)
            p = QPainter(out)
            p.drawPixmap(0, 0, scaled)
            p.drawImage(0, 0, ov)
            p.end()
            self.view.setPixmap(out)
        else:
            self.view.setPixmap(scaled)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if hasattr(self, "_qimg"):
            self._set_display(self._qimg)

    def eventFilter(self, obj, ev):
        if obj is self.view and ev.type() == QEvent.Type.MouseButtonPress and hasattr(self, "_bgr") and self._bgr is not None:
            # 轉換座標到原圖
            pos = ev.position().toPoint()
            label_size = self.view.size()
            view_w = max(1, label_size.width() - 16)
            view_h = max(1, label_size.height() - 16)
            h, w = self._bgr.shape[:2]
            k = min(view_w / w, view_h / h)
            disp_w, disp_h = int(w * k), int(h * k)
            off_x = (label_size.width() - disp_w) // 2
            off_y = (label_size.height() - disp_h) // 2
            x = pos.x() - off_x; y = pos.y() - off_y
            if 0 <= x < disp_w and 0 <= y < disp_h:
                sx, sy = int(x / k), int(y / k)
                self._seed = (sx, sy)
                self.info.setText(f"Seed: ({sx},{sy})  H={self.sl_h.value()} S={self.sl_s.value()} V={self.sl_v.value()}")
        return super().eventFilter(obj, ev)

    def _update_hsv_preview(self):
        if not hasattr(self, "_bgr") or self._bgr is None:
            return
        from core.hsv_bg import compute_alpha
        s = self.sl_strength.value() / 100.0
        opts = {
            "tol_h": int(self.sl_h.value()),
            "tol_s": int(self.sl_s.value()),
            "tol_v": int(self.sl_v.value()),
            "strength": float(s),
            "erode_iter": 1,
            "dilate_iter": 0,
            "feather_px": 2.0,
        }
        try:
            alpha = compute_alpha(self._bgr.copy(), opts)
            h, w = alpha.shape[:2]
            ov = np.zeros((h, w, 4), dtype=np.uint8)
            ov[..., 1] = 255
            ov[..., 3] = (alpha > 0).astype(np.uint8) * 90
            qov = QImage(ov.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
            self._set_display(self._qimg, qov)
        except Exception:
            self._set_display(self._qimg)

    def _on_reload(self):
        # 若外部提供 refresh_fn，沿用舊行為；否則依目前模式重載當前幀
        if self._refresh_fn is not None:
            new_path = self._refresh_fn() or self._img_path
            self._img_path = new_path
            self._src_path = new_path
            # 切回圖片模式
            self._is_video = False
            self._is_animated_image = False
            # 隱藏幀 UI
            if hasattr(self, 'frame_slider'):
                self.frame_slider.setVisible(False)
            if hasattr(self, 'frame_label'):
                self.frame_label.setVisible(False)
            self._load(self._img_path)
        else:
            self._load_current()

    def _on_apply_hsv(self):
        s = self.sl_strength.value() / 100.0
        payload = {"tol_h": int(self.sl_h.value()), "tol_s": int(self.sl_s.value()), "tol_v": int(self.sl_v.value()), "strength": float(s)}
        self.hsvChanged.emit(payload)

    def _on_use_seed(self):
        if self._seed is not None:
            self.seedSelected.emit(self._seed)

    # ---------- 影片/動圖幀選擇支援 ----------
    def _init_media(self):
        path = self._src_path
        ext = os.path.splitext(path)[1].lower()
        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        try:
            if ext in video_exts:
                self._is_video = True
                self._cap = cv2.VideoCapture(path)
                if not self._cap.isOpened():
                    raise RuntimeError("cannot open video")
                frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
                self._total_frames = max(1, frames)
                self._fps = fps if fps > 0 else 15.0
                if self._init_ms is not None and self._fps > 0:
                    self._current_frame = min(self._total_frames-1, max(0, int(round(self._init_ms * self._fps / 1000.0))))
                self._setup_frame_ui(True)
            else:
                r = QImageReader(path)
                r.setDecideFormatFromContent(True)
                self._reader = r
                if r.supportsAnimation():
                    self._is_animated_image = True
                    cnt = r.imageCount()
                    self._total_frames = cnt if cnt and cnt > 0 else 1
                    self._fps = 0.0
                    self._setup_frame_ui(True)
                else:
                    self._setup_frame_ui(False)
        except Exception:
            self._is_video = False
            self._is_animated_image = False
            self._setup_frame_ui(False)

    def _setup_frame_ui(self, show: bool):
        self.frame_slider.setVisible(show)
        self.frame_label.setVisible(show)
        if show:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setRange(0, max(0, self._total_frames - 1))
            self.frame_slider.setValue(int(self._current_frame))
            self.frame_slider.blockSignals(False)
            self._update_frame_label()
        else:
            self.frame_label.setText("Frame: -/-  (00:00)")

    def _hhmmss(self, seconds: float) -> str:
        seconds = max(0, int(seconds))
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _update_frame_label(self):
        if self._is_video and self._fps > 0:
            t = self._current_frame / self._fps
            self.frame_label.setText(f"Frame: {self._current_frame+1}/{self._total_frames}  ({self._hhmmss(t)})")
        elif self._is_animated_image:
            self.frame_label.setText(f"Frame: {self._current_frame+1}/{self._total_frames}")
        else:
            self.frame_label.setText("Frame: -/-  (00:00)")

    def _on_seek_frame(self, v: int):
        self._current_frame = int(v)
        self._update_frame_label()
        try:
            self.info.setText("Loading…")
        except Exception:
            pass
        self._load_current()

    def _load_current(self):
        if self._is_video:
            self._bgr = self._read_video_frame(self._current_frame)
        elif self._is_animated_image:
            self._bgr = self._read_animated_image_frame(self._current_frame)
        else:
            self._bgr = cv2.imread(self._src_path, cv2.IMREAD_COLOR)
        if self._bgr is None:
            self.view.setText("讀取影像失敗")
            return
        rgb = cv2.cvtColor(self._bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self._qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._set_display(self._qimg)
        self._update_hsv_preview()

    def _read_video_frame(self, index: int):
        try:
            key = ("vid", int(index))
            c = self._cache_get(key)
            if c is not None:
                return c
            if self._cap is None or not self._cap.isOpened():
                return None
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(index))
            ok, frame = self._cap.read()
            if ok and frame is not None:
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                self._cache_put(key, frame)
                return frame
            return None
        except Exception:
            return None

    def _read_animated_image_frame(self, index: int):
        try:
            r = self._reader
            if r is None:
                return None
            key = ("anim", int(index))
            c = self._cache_get(key)
            if c is not None:
                return c
            try:
                if hasattr(r, 'jumpToImage') and r.jumpToImage(int(index)):
                    pass
                elif hasattr(r, 'setCurrentImageNumber'):
                    r.setCurrentImageNumber(int(index))
            except Exception:
                pass
            img = r.read()
            if img is None or img.isNull():
                return None
            img = img.convertToFormat(QImage.Format.Format_RGB888)
            w = img.width(); h = img.height()
            ptr = img.bits(); ptr.setsize(h * w * 3)
            arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 3))
            bgr = arr[:, :, ::-1].copy()
            self._cache_put(key, bgr)
            return bgr
        except Exception:
            return None

    # ----- LRU cache helpers -----
    def _cache_get(self, key: tuple):
        try:
            if key in self._frame_cache:
                val = self._frame_cache.pop(key)
                self._frame_cache[key] = val
                return val
        except Exception:
            return None
        return None

    def _cache_put(self, key: tuple, frame: np.ndarray):
        try:
            if key in self._frame_cache:
                self._frame_cache.pop(key)
            self._frame_cache[key] = frame
            while len(self._frame_cache) > int(self._cache_cap):
                self._frame_cache.popitem(last=False)
        except Exception:
            pass
