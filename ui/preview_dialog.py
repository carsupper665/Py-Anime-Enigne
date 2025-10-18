from __future__ import annotations
from typing import Optional, Callable
from PyQt6.QtCore import Qt, pyqtSignal, QEvent
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QSlider, QCheckBox
from PyQt6.QtGui import QPixmap, QImage, QPainter
import numpy as np
import cv2


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
                 init_hsv: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("影格預覽")
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("QWidget { background:#222; color:#fff; font-family:'Source Han Sans TC'; }")

        self._img_path = img_path
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

        self._load(self._img_path)

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
        if self._refresh_fn is None:
            return
        new_path = self._refresh_fn() or self._img_path
        self._img_path = new_path
        self._load(new_path)

    def _on_apply_hsv(self):
        s = self.sl_strength.value() / 100.0
        payload = {"tol_h": int(self.sl_h.value()), "tol_s": int(self.sl_s.value()), "tol_v": int(self.sl_v.value()), "strength": float(s)}
        self.hsvChanged.emit(payload)

    def _on_use_seed(self):
        if self._seed is not None:
            self.seedSelected.emit(self._seed)
