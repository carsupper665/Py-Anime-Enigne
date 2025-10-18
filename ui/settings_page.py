from __future__ import annotations
import os
from typing import Dict, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFormLayout, QComboBox, QHBoxLayout,
    QLineEdit, QPushButton, QFileDialog
)
from PyQt6.QtCore import pyqtSignal, Qt

from core.config import load_config, save_config, get_default_config


class SettingsPage(QWidget):
    configSaved = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SettingsPage")
        self._cfg: Dict[str, Any] = load_config()

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)
        # 全域樣式（字體與文字顏色）
        self.setStyleSheet(
            """
            QWidget { font-family: 'Source Han Sans TC'; color: #FFFFFF; }
            QLineEdit, QComboBox { background-color: #1b1b1b; color: #FFFFFF; border: 1px solid #333; border-radius: 6px; padding: 4px 6px; }
            QLabel { color: #FFFFFF; }
            """
        )

        title = QLabel("設定", self)
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        title.setStyleSheet(
            "background:none; font-family:'Source Han Sans TC'; font-weight:600; font-size:26px; color:#FFFFFF;"
        )
        root.addWidget(title)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        # Output dir
        od_row = QHBoxLayout()
        self.out_dir = QLineEdit(self)
        self.btn_browse_dir = QPushButton("瀏覽", self)
        self.btn_browse_dir.clicked.connect(self._pick_dir)
        od_row.addWidget(self.out_dir)
        od_row.addWidget(self.btn_browse_dir)

        # Output format
        self.fmt_image = QComboBox(self)
        self.fmt_image.addItems(["webp", "png"])
        self.fmt_anim = QComboBox(self)
        self.fmt_anim.addItems(["webp", "gif"])

        # OpenVINO model path
        ov_row = QHBoxLayout()
        self.ov_path = QLineEdit(self)
        self.btn_browse_ov = QPushButton("選擇", self)
        self.btn_browse_ov.clicked.connect(self._pick_model)
        ov_row.addWidget(self.ov_path)
        ov_row.addWidget(self.btn_browse_ov)

        form.addRow("輸出資料夾", od_row)
        form.addRow("靜態輸出格式", self.fmt_image)
        form.addRow("動圖輸出格式", self.fmt_anim)
        form.addRow("OpenVINO 模型", ov_row)

        # HSV 詳細設定
        hsv_form = QFormLayout()
        self.h_tol_h = QLineEdit(self); self.h_tol_h.setPlaceholderText("10")
        self.h_tol_s = QLineEdit(self); self.h_tol_s.setPlaceholderText("60")
        self.h_tol_v = QLineEdit(self); self.h_tol_v.setPlaceholderText("60")
        self.h_strength = QLineEdit(self); self.h_strength.setPlaceholderText("1.5")
        self.h_erode = QLineEdit(self); self.h_erode.setPlaceholderText("1")
        self.h_dilate = QLineEdit(self); self.h_dilate.setPlaceholderText("0")
        self.h_feather = QLineEdit(self); self.h_feather.setPlaceholderText("2.0")
        self.h_guided = QComboBox(self); self.h_guided.addItems(["false", "true"])

        hsv_form.addRow("HSV 容差 H", self.h_tol_h)
        hsv_form.addRow("HSV 容差 S", self.h_tol_s)
        hsv_form.addRow("HSV 容差 V", self.h_tol_v)
        hsv_form.addRow("強度倍率", self.h_strength)
        hsv_form.addRow("侵蝕次數", self.h_erode)
        hsv_form.addRow("膨脹次數", self.h_dilate)
        hsv_form.addRow("羽化像素", self.h_feather)
        hsv_form.addRow("導向濾波", self.h_guided)

        root.addLayout(form)
        root.addLayout(hsv_form)

        btns = QHBoxLayout()
        self.btn_apply = QPushButton("套用", self)
        self.btn_reset = QPushButton("還原預設", self)
        self.btn_apply.clicked.connect(self._apply)
        self.btn_reset.clicked.connect(self._reset)
        for b in (self.btn_apply, self.btn_reset):
            b.setStyleSheet("QPushButton{background:#333;color:#EEE;border:none;border-radius:6px;padding:6px 12px;} QPushButton:hover{background:#444}")
        btns.addStretch(1)
        btns.addWidget(self.btn_reset)
        btns.addWidget(self.btn_apply)
        root.addLayout(btns)

        self._load_to_ui(self._cfg)

    def _load_to_ui(self, cfg: Dict[str, Any]):
        self.out_dir.setText(cfg.get("output", {}).get("dir", "./animes"))
        self.fmt_image.setCurrentText(cfg.get("output", {}).get("image", "webp"))
        self.fmt_anim.setCurrentText(cfg.get("output", {}).get("anim", "webp"))
        self.ov_path.setText(cfg.get("openvino", {}).get("model_path", ""))
        hsv = cfg.get("hsv", {})
        self.h_tol_h.setText(str(hsv.get("tol_h", 10)))
        self.h_tol_s.setText(str(hsv.get("tol_s", 60)))
        self.h_tol_v.setText(str(hsv.get("tol_v", 60)))
        self.h_strength.setText(str(hsv.get("strength", 1.5)))
        self.h_erode.setText(str(hsv.get("erode_iter", 1)))
        self.h_dilate.setText(str(hsv.get("dilate_iter", 0)))
        self.h_feather.setText(str(hsv.get("feather_px", 2.0)))
        self.h_guided.setCurrentText("true" if hsv.get("use_guided", False) else "false")

    def _apply(self):
        self._cfg.setdefault("output", {})
        self._cfg["output"]["dir"] = self.out_dir.text().strip() or "./animes"
        self._cfg["output"]["image"] = self.fmt_image.currentText()
        self._cfg["output"]["anim"] = self.fmt_anim.currentText()
        self._cfg.setdefault("openvino", {})
        self._cfg["openvino"]["model_path"] = self.ov_path.text().strip()
        # HSV 設定
        self._cfg.setdefault("hsv", {})
        h = self._cfg["hsv"]
        def _num(x, ty, default):
            try: return ty(x)
            except Exception: return default
        h["tol_h"] = _num(self.h_tol_h.text(), int, 10)
        h["tol_s"] = _num(self.h_tol_s.text(), int, 60)
        h["tol_v"] = _num(self.h_tol_v.text(), int, 60)
        h["strength"] = _num(self.h_strength.text(), float, 1.5)
        h["erode_iter"] = _num(self.h_erode.text(), int, 1)
        h["dilate_iter"] = _num(self.h_dilate.text(), int, 0)
        h["feather_px"] = _num(self.h_feather.text(), float, 2.0)
        h["use_guided"] = (self.h_guided.currentText() == "true")
        save_config(self._cfg)
        self.configSaved.emit(dict(self._cfg))

    def _reset(self):
        self._cfg = get_default_config()
        self._load_to_ui(self._cfg)

    def _pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "選擇輸出資料夾", self.out_dir.text() or os.getcwd())
        if d:
            self.out_dir.setText(d)

    def _pick_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "選擇 OpenVINO 模型", os.getcwd(), "Model (*.onnx *.xml)")
        if p:
            self.ov_path.setText(p)
