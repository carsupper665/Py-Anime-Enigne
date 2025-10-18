from __future__ import annotations
import os
from typing import Optional, Tuple

from PyQt6.QtCore import Qt, QEvent, QSize
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QHBoxLayout, QSlider, QPushButton
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor

import cv2
import numpy as np


class WandDialog(QDialog):
    """魔術棒選取對話框
    - 顯示圖像，點擊以選取 seed（座標）
    - 調整 Hue 容差（選取範圍）
    - 即時預覽疊色區域
    回傳：seed(tuple)、tolH(int)
    """

    def __init__(self, img_path: str, tol_h: int = 10, parent=None):
        super().__init__(parent)
        self.setWindowTitle("魔術棒選取")
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            """
            QWidget { background:#222; color:#fff; font-family:'Source Han Sans TC'; }
            QSlider::groove:horizontal { height:6px; background:#444; }
            QSlider::handle:horizontal { width:12px; background:#2a7ae2; margin:-6px 0; border-radius:6px; }
            QPushButton { background:#333; color:#EEE; border:none; border-radius:6px; padding:6px 12px; }
            QPushButton:hover { background:#444; }
            """
        )

        self.img_path = img_path
        self.seed: Optional[Tuple[int, int]] = None
        self.tol_h = max(1, min(60, int(tol_h)))

        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        self.info = QLabel("點擊圖片以選取顏色種子")
        v.addWidget(self.info)

        # 預覽圖
        self.view = QLabel()
        self.view.setMinimumSize(480, 270)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setStyleSheet("QLabel { background:#111; border:1px solid #333; border-radius:8px; }")
        v.addWidget(self.view, 1)

        # 控制列
        row = QHBoxLayout()
        self.lab_tol = QLabel(f"容差(H): {self.tol_h}")
        self.sl_tol = QSlider(Qt.Orientation.Horizontal)
        self.sl_tol.setMinimum(1); self.sl_tol.setMaximum(60); self.sl_tol.setValue(self.tol_h)
        self.sl_tol.valueChanged.connect(self._on_tol_changed)
        row.addWidget(self.lab_tol)
        row.addWidget(self.sl_tol)
        v.addLayout(row)

        # 操作按鈕
        btns = QHBoxLayout(); btns.addStretch(1)
        self.btn_cancel = QPushButton("取消")
        self.btn_ok = QPushButton("確定")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self._accept)
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_ok)
        v.addLayout(btns)

        # 載入圖像
        self._load_image()
        self.view.installEventFilter(self)

    # 影像載入與縮放/座標換算
    def _load_image(self):
        bgr = cv2.imread(self.img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            self._rgb = None
            self._src_wh = (0, 0)
            self.view.setText("讀取圖片失敗")
            return
        self._bgr = bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._rgb = rgb
        self._src_wh = (rgb.shape[1], rgb.shape[0])
        self._draw_base()

    def _draw_base(self):
        if self._rgb is None:
            return
        h, w = self._rgb.shape[:2]
        qimg = QImage(self._rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self._base_pix = pix
        self._set_display_pix(pix)

    def _set_display_pix(self, pix: QPixmap, overlay: QImage | None = None):
        # 將 pix 按 view 大小等比縮放；若有 overlay，疊加
        scaled = pix.scaled(self.view.size() - QSize(16, 16), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        if overlay is not None:
            # overlay 需與 scaled 同尺寸
            ov = QImage(overlay)
            ov = ov.scaled(scaled.size(), Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
            out = QPixmap(scaled.size())
            out.fill(Qt.GlobalColor.transparent)
            p = QPainter(out)
            p.drawPixmap(0, 0, scaled)
            p.drawImage(0, 0, ov)
            p.end()
            self.view.setPixmap(out)
        else:
            self.view.setPixmap(scaled)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if hasattr(self, "_base_pix"):
            self._set_display_pix(self._base_pix)

    def eventFilter(self, obj, ev):
        if obj is self.view and ev.type() == QEvent.Type.MouseButtonPress and self._rgb is not None:
            pos = ev.position().toPoint()
            mapped = self._map_to_source(pos)
            if mapped is not None:
                self.seed = mapped
                self._update_preview()
        return super().eventFilter(obj, ev)

    def _on_tol_changed(self, v: int):
        self.tol_h = v
        self.lab_tol.setText(f"容差(H): {v}")
        if self.seed is not None:
            self._update_preview()

    def _map_to_source(self, pt):
        # 計算 view 中可見區域與縮放比
        if self._rgb is None:
            return None
        label_size = self.view.size()
        view_w = max(1, label_size.width() - 16)
        view_h = max(1, label_size.height() - 16)
        src_w, src_h = self._src_wh
        k = min(view_w / src_w, view_h / src_h)
        disp_w, disp_h = int(src_w * k), int(src_h * k)
        off_x = (label_size.width() - disp_w) // 2
        off_y = (label_size.height() - disp_h) // 2
        x = pt.x() - off_x
        y = pt.y() - off_y
        if x < 0 or y < 0 or x >= disp_w or y >= disp_h:
            return None
        src_x = int(x / k)
        src_y = int(y / k)
        return (src_x, src_y)

    def _update_preview(self):
        # 計算遮罩並疊色顯示
        if self._rgb is None or self.seed is None:
            return
        from core.wand import compute_mask
        bgr = self._bgr
        h, w = bgr.shape[:2]
        mask = compute_mask(bgr, self.seed, {"tolH": int(self.tol_h), "tolS": 60, "tolV": 60, "contiguous": True, "use_edge_barrier": True, "connectivity": 8})

        # downscale mask to display size
        disp = self.view.pixmap()
        if disp is None:
            self._set_display_pix(self._base_pix)
            disp = self.view.pixmap()
        if disp is None:
            return

        # 產生綠色半透明 overlay
        kx = disp.width() / w
        ky = disp.height() / h
        # 直接轉成 QImage 讓 _set_display_pix 做縮放匹配
        ov = np.zeros((h, w, 4), dtype=np.uint8)
        ov[..., 1] = 255  # G
        ov[..., 3] = (mask > 0).astype(np.uint8) * 90
        qov = QImage(ov.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)

        # 更新資訊
        x, y = self.seed
        self.info.setText(f"座標: ({x}, {y})  容差(H): {self.tol_h}")
        self._set_display_pix(self._base_pix, qov)

    def _accept(self):
        if self.seed is None:
            # 沒有選取也允許回傳 None，外側可提示
            self.reject()
            return
        self.accept()

    def get_result(self) -> Optional[tuple[tuple[int, int], int]]:
        if self.result() == QDialog.DialogCode.Accepted and self.seed is not None:
            return (self.seed, int(self.tol_h))
        return None
