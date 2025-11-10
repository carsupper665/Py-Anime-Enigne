from __future__ import annotations
from typing import Optional
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QCheckBox,
    QPushButton,
)


class SaveDialog(QDialog):
    """命名 + Animated WebP 參數對話框。

    - 通用參數（寫回 config）: quality, max_fps, loop
    - 非通用（僅當次）: target_width/target_height（可選）
    """

    def __init__(
        self,
        parent=None,
        *,
        default_name: str = "output.webp",
        quality: int = 75,
        max_fps: int = 15,
        auto_fps: bool = False,
        loop: bool = True,
        enable_size: bool = False,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("儲存設定")
        self.setModal(True)

        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        # 檔名
        row_name = QHBoxLayout()
        row_name.addWidget(QLabel("檔名"))
        self.ed_name = QLineEdit(default_name)
        row_name.addWidget(self.ed_name)
        v.addLayout(row_name)

        # Quality
        row_q = QHBoxLayout()
        row_q.addWidget(QLabel("品質 (1-100)"))
        self.sb_quality = QSpinBox()
        self.sb_quality.setRange(1, 100)
        self.sb_quality.setValue(int(quality))
        row_q.addWidget(self.sb_quality)
        v.addLayout(row_q)

        # Max FPS（支援自動）
        row_fps = QHBoxLayout()
        row_fps.addWidget(QLabel("最大 FPS"))
        self.cb_auto_fps = QCheckBox("自動")
        self.cb_auto_fps.setChecked(bool(auto_fps))
        self.sb_maxfps = QSpinBox()
        self.sb_maxfps.setRange(1, 120)
        self.sb_maxfps.setValue(int(max_fps))
        self.sb_maxfps.setEnabled(not bool(auto_fps))

        def _on_auto(v: bool):
            try:
                self.sb_maxfps.setEnabled(not bool(v))
            except Exception:
                pass

        self.cb_auto_fps.toggled.connect(_on_auto)
        row_fps.addWidget(self.cb_auto_fps)
        row_fps.addWidget(self.sb_maxfps)
        v.addLayout(row_fps)

        # Loop
        self.cb_loop = QCheckBox("無限循環")
        self.cb_loop.setChecked(bool(loop))
        v.addWidget(self.cb_loop)

        # 非通用尺寸（可選）
        self._size_enabled = bool(enable_size)
        row_sz = QHBoxLayout()
        row_sz.addWidget(QLabel("目標尺寸 (可選)"))
        self.sb_w = QSpinBox()
        self.sb_w.setRange(1, 8192)
        self.sb_h = QSpinBox()
        self.sb_h.setRange(1, 8192)
        if target_width:
            self.sb_w.setValue(int(target_width))
        if target_height:
            self.sb_h.setValue(int(target_height))
        for w in (self.sb_w, self.sb_h):
            w.setEnabled(self._size_enabled)
        row_sz.addWidget(QLabel("W"))
        row_sz.addWidget(self.sb_w)
        row_sz.addWidget(QLabel("H"))
        row_sz.addWidget(self.sb_h)
        v.addLayout(row_sz)

        # Buttons
        row_btn = QHBoxLayout()
        row_btn.addStretch(1)
        ok = QPushButton("確定")
        cancel = QPushButton("取消")
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        row_btn.addWidget(ok)
        row_btn.addWidget(cancel)
        v.addLayout(row_btn)

    def result_payload(self) -> dict:
        name = self.ed_name.text().strip()
        q = int(self.sb_quality.value())
        auto = (
            bool(self.cb_auto_fps.isChecked())
            if hasattr(self, "cb_auto_fps")
            else False
        )
        fps = 0 if auto else int(self.sb_maxfps.value())
        loop = bool(self.cb_loop.isChecked())
        payload = {
            "name": name,
            "quality": q,
            "max_fps": fps,
            "max_fps_auto": auto,
            "loop": loop,
        }
        if self._size_enabled:
            w = int(self.sb_w.value())
            h = int(self.sb_h.value())
            if w > 0 and h > 0:
                payload["target_size"] = {"w": w, "h": h}
        return payload
