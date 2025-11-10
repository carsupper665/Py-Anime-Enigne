from __future__ import annotations
import os
from typing import Optional, Callable

from PyQt6.QtCore import Qt, pyqtSignal, QEvent
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QSlider,
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QImageReader

from core.export_context import ExportOptions
from ui.services.preview_service import PreviewService, PreviewFrame

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


class PreviewImageDialog(QDialog):
    hsvChanged = pyqtSignal(dict)
    seedSelected = pyqtSignal(object)

    def __init__(
        self,
        img_path: str,
        refresh_fn: Optional[Callable[[], Optional[str]]] = None,
        init_hsv: Optional[dict] = None,
        init_ms: Optional[int] = None,
        export_options: Optional[ExportOptions] = None,
        service: Optional[PreviewService] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("影格預覽")
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            "QWidget { background:#222; color:#fff; font-family:'Source Han Sans TC'; }"
        )

        self._path = img_path
        self._refresh_fn = refresh_fn
        self._service = service or PreviewService()
        self._export_options = export_options
        self._current_preview: Optional[PreviewFrame] = None

        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        self.info = QLabel("點擊圖片以取樣（魔術棒 seed）/ 調整 HSV 預覽")
        v.addWidget(self.info)

        self.view = QLabel()
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumSize(640, 360)
        self.view.setStyleSheet(
            "QLabel { background:#111; border:1px solid #333; border-radius:8px; }"
        )
        v.addWidget(self.view, 1)
        self.view.installEventFilter(self)

        row_f = QHBoxLayout()
        self.frame_label = QLabel("Frame: -/-  (00:00)")
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        row_f.addWidget(self.frame_label)
        row_f.addWidget(self.frame_slider)
        v.addLayout(row_f)

        row1 = QHBoxLayout()
        self.sl_h = QSlider(Qt.Orientation.Horizontal)
        self.sl_h.setRange(1, 60)
        self.sl_s = QSlider(Qt.Orientation.Horizontal)
        self.sl_s.setRange(1, 100)
        self.sl_v = QSlider(Qt.Orientation.Horizontal)
        self.sl_v.setRange(1, 100)
        row1.addWidget(QLabel("H"))
        row1.addWidget(self.sl_h)
        row1.addWidget(QLabel("S"))
        row1.addWidget(self.sl_s)
        row1.addWidget(QLabel("V"))
        row1.addWidget(self.sl_v)
        v.addLayout(row1)

        row2 = QHBoxLayout()
        self.sl_strength = QSlider(Qt.Orientation.Horizontal)
        self.sl_strength.setRange(50, 300)
        row2.addWidget(QLabel("倍率"))
        row2.addWidget(self.sl_strength)
        v.addLayout(row2)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.btn_reload = QPushButton("重新擷取")
        self.btn_apply_hsv = QPushButton("套用到控制")
        self.btn_use_seed = QPushButton("使用取樣座標")
        self.btn_close = QPushButton("關閉")
        for b in (
            self.btn_reload,
            self.btn_apply_hsv,
            self.btn_use_seed,
            self.btn_close,
        ):
            b.setStyleSheet(
                "QPushButton { background:#333; color:#EEE; border:none; border-radius:6px; padding:6px 12px; }"
            )
            btns.addWidget(b)
        v.addLayout(btns)

        self.btn_reload.clicked.connect(self._on_reload)
        self.btn_apply_hsv.clicked.connect(self._on_apply_hsv)
        self.btn_use_seed.clicked.connect(self._on_use_seed)
        self.btn_close.clicked.connect(self.accept)

        for slider, key in (
            (self.sl_h, "tol_h"),
            (self.sl_s, "tol_s"),
            (self.sl_v, "tol_v"),
        ):
            slider.valueChanged.connect(
                lambda value, k=key: self._on_hsv_change(k, value)
            )
        self.sl_strength.valueChanged.connect(
            lambda value: self._on_hsv_change("strength", value / 100.0)
        )
        self.frame_slider.valueChanged.connect(self._on_seek_frame)

        self._prepare_source(img_path)
        if init_hsv:
            for key, value in init_hsv.items():
                if key in self._service.get_state().hsv:
                    self._service.update_hsv(key, float(value))
        self._apply_hsv_to_sliders()

        if init_ms and self._service.get_source_info()["fps"] > 0:
            fps = self._service.get_source_info()["fps"]
            frame_index = int(max(0, init_ms / 1000.0) * fps)
            self._service.seek(frame_index)

        self._render_current_frame()

    # --- Source preparation ---
    def _prepare_source(self, path: str) -> None:
        ext = os.path.splitext(path)[1].lower()
        if ext in VIDEO_EXTS:
            self._service.set_video_source(path)
        else:
            reader = QImageReader(path)
            reader.setDecideFormatFromContent(True)
            if reader.supportsAnimation():
                self._service.set_animated_source(path)
            else:
                self._service.set_static_source(path)
        self._sync_slider()

    def _sync_slider(self) -> None:
        info = self._service.get_source_info()
        total = max(1, int(info["total_frames"]))
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, total - 1)
        self.frame_slider.setValue(int(self._service.get_state().current_frame))
        self.frame_slider.blockSignals(False)
        show_slider = total > 1
        self.frame_slider.setVisible(show_slider)
        self.frame_label.setVisible(show_slider)
        self._update_frame_label()

    # --- Rendering ---
    def _render_current_frame(self) -> None:
        try:
            frame = self._service.load_current_frame(apply_hsv=True)
        except Exception as exc:
            self.view.setText(str(exc))
            return
        self._current_preview = frame
        self._display_frame(frame)
        self._update_frame_label()

    def _display_frame(self, frame: PreviewFrame) -> None:
        pixmap = QPixmap.fromImage(frame.image)
        scaled = pixmap.scaled(
            self.view.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if frame.overlay is not None:
            overlay = frame.overlay.scaled(
                scaled.size(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            canvas = QPixmap(scaled.size())
            canvas.fill(Qt.GlobalColor.transparent)
            painter = QPainter(canvas)
            painter.drawPixmap(0, 0, scaled)
            painter.drawImage(0, 0, overlay)
            painter.end()
            self.view.setPixmap(canvas)
        else:
            self.view.setPixmap(scaled)

    def _update_frame_label(self) -> None:
        info = self._service.get_source_info()
        state = self._service.get_state()
        if info["is_video"] and info["fps"] > 0:
            seconds = state.current_frame / info["fps"]
            self.frame_label.setText(
                f"Frame: {state.current_frame + 1}/{max(1, info['total_frames'])}  ({self._format_time(seconds)})"
            )
        elif info["is_animated"]:
            self.frame_label.setText(
                f"Frame: {state.current_frame + 1}/{max(1, info['total_frames'])}"
            )
        else:
            self.frame_label.setText("Frame: -/-  (00:00)")

    # --- HSV handlers ---
    def _apply_hsv_to_sliders(self) -> None:
        state = self._service.get_state()
        self.sl_h.setValue(int(state.hsv["tol_h"]))
        self.sl_s.setValue(int(state.hsv["tol_s"]))
        self.sl_v.setValue(int(state.hsv["tol_v"]))
        self.sl_strength.setValue(int(float(state.hsv["strength"]) * 100))

    def _on_hsv_change(self, key: str, value: float) -> None:
        self._service.update_hsv(key, value)
        self._render_current_frame()

    # --- Slots ---
    def _on_seek_frame(self, index: int) -> None:
        self._service.seek(int(index))
        self._render_current_frame()

    def _on_reload(self) -> None:
        if not self._refresh_fn:
            return
        new_path = self._refresh_fn()
        if not new_path:
            return
        self._path = new_path
        self._service.clear_cache()
        self._prepare_source(new_path)
        self._render_current_frame()

    def _on_apply_hsv(self) -> None:
        self.hsvChanged.emit(dict(self._service.get_state().hsv))

    def _on_use_seed(self) -> None:
        self.seedSelected.emit(self._service.get_state().seed)

    # --- Utilities ---
    def _format_time(self, seconds: float) -> str:
        seconds = max(0, int(seconds))
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def eventFilter(self, obj, ev):
        if (
            obj is self.view
            and ev.type() == QEvent.Type.MouseButtonPress
            and self._current_preview is not None
        ):
            pos = ev.position().toPoint()
            mapped = self._map_click_to_image(pos)
            if mapped is not None:
                self._service.update_seed(mapped)
                self.seedSelected.emit(mapped)
                try:
                    self.info.setText(
                        f"Seed: {mapped}  H={self.sl_h.value()} S={self.sl_s.value()} V={self.sl_v.value()}"
                    )
                except Exception:
                    pass
                self._render_current_frame()
        return super().eventFilter(obj, ev)

    def _map_click_to_image(self, pt):
        if self._current_preview is None:
            return None
        qimg = self._current_preview.image
        label_size = self.view.size()
        view_w = max(1, label_size.width() - 16)
        view_h = max(1, label_size.height() - 16)
        src_w, src_h = qimg.width(), qimg.height()
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
