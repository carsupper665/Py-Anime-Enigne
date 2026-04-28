# ui/home_page.py
from typing import Any, Dict, Optional, Tuple

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QFileDialog,
    QSplitter,
    QInputDialog,
    QMessageBox,
    QSlider,
    QStackedLayout,
    QWidget as QW,
    QCheckBox,
    QComboBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QUrl, QEvent
from PyQt6.QtGui import QPixmap, QImageReader, QDragEnterEvent, QDropEvent, QMovie

try:
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput  # 需要系統多媒體相依
    from PyQt6.QtMultimediaWidgets import QVideoWidget

    _MULTIMEDIA_AVAILABLE = True
except Exception:
    # 在缺少 libpulse/gstreamer（WSL/最小化容器）時，允許應用啟動但關閉影片功能
    QMediaPlayer = None  # type: ignore
    QAudioOutput = None  # type: ignore
    QVideoWidget = None  # type: ignore
    _MULTIMEDIA_AVAILABLE = False
import logging
import os, shutil, subprocess
from core.config import save_config
from core.export_context import ExportOptions, merge_export_settings
from core.job_models import JobRequest
from core.export.services import (
    ExportCommandBuilder,
    TempDirectoryManager,
    VideoExportService,
)
from core.diagnostics import attach_diagnostic, generate_diagnostic_id, log_structured
from ui.services.preview_service import PreviewService
from ui.services.save_controller import SaveDialogResult, SaveExportController, SavePlan

_BTN_STYLE = """
    QPushButton {
        background-color: #333; color: #EEE; border: none; border-radius: 6px;
        padding: 6px 12px; width: 80px; font-size: 14px; font-family: "Source Han Sans TC"
    }
    QPushButton:hover { background-color: #444; }
    QPushButton:pressed { background-color: #555; }
"""
_SAVE_STYLE = """
    QPushButton {
        background-color: #2a7ae2; color: #EEE; border: none; border-radius: 6px;
        padding: 6px 12px; width: 80px; font-size: 14px; font-family: "Source Han Sans TC"
    }
    QPushButton:hover { background-color: #4a90e2; }
    QPushButton:pressed { background-color: #357ae8; }
    QPushButton:disabled { background-color: #555; color: #888; }
"""
_BASE = """
font-weight: 300; font-size: 14px; color: #FFFFFF; font-family: "Source Han Sans TC";
"""

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
# 供缺少多媒體相依時提示用
_ALL_VIDEO_EXTS = set(VIDEO_EXTS)
_EXTS = ["gif", "webp", "png"]


class HomePage(QWidget):
    update_data = pyqtSignal()
    on_exception = pyqtSignal(object)
    fileSelected = pyqtSignal(str)
    toast = pyqtSignal(dict)  # {level,title,message,duration}
    # src_path, prefer, options
    removeBg = pyqtSignal(str, str, dict)
    # 佇列：加入任務
    enqueueJob = pyqtSignal(str, str, dict)

    def __init__(self, parent):
        super().__init__(parent)
        self.p = parent
        self.logger = getattr(parent, "logger", logging.getLogger(__name__))
        self.setObjectName("HomePage")
        self._movie: QMovie | None = None
        self._current_path: str | None = None
        self.setAcceptDrops(True)
        self.setStyleSheet(_BASE)
        self._command_builder = ExportCommandBuilder(logger=self.logger)
        self._temp_manager = TempDirectoryManager(logger=self.logger)
        self._video_service = VideoExportService(
            runner=self._run_ffmpeg_command,
            command_builder=self._command_builder,
            logger=self.logger,
        )
        self._last_export_options = ExportOptions()
        self._preview_service = PreviewService()
        self._preview_source_path: Optional[str] = None
        self._save_controller = SaveExportController(
            logger=self.logger,
            config_provider=lambda: self.p.config if hasattr(self.p, "config") else {},
            config_saver=save_config,
            enqueue_job=self._enqueue_job_request,
            toast_emitter=self.toast.emit,
            get_out_dir=self._get_out_dir,
            confirm_overwrite=self._confirm_overwrite,
            show_message=self._show_message,
            can_direct_copy=self._can_direct_copy,
            build_queue_job=self._build_queue_job,
            on_options_updated=self._set_last_export_options,
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Anime Engine", self)
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        title.setStyleSheet(
            """
            background-color: none; font-family: "Source Han Sans TC";
            font-weight: 600; font-size: 30px; color: #FFFFFF;
            """
        )

        # 左側：圖片/GIF 與 影片堆疊
        self.preview_img = QLabel(self)
        self.preview_img.setMinimumSize(480, 270)
        self.preview_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_img.setStyleSheet(
            """
            QLabel { background-color: #111; border: 1px solid #333; border-radius: 12px; }
            """
        )

        self.left_stack = QStackedLayout()
        self.left_stack.addWidget(self.preview_img)  # 0 = 圖片/GIF

        # 僅在多媒體相依可用時啟用影片播放功能
        if _MULTIMEDIA_AVAILABLE:
            self.video_widget = QVideoWidget(self)
            self.player = QMediaPlayer(self)
            self.audio_out = QAudioOutput(self)
            self.player.setAudioOutput(self.audio_out)
            self.player.setVideoOutput(self.video_widget)
            self.left_stack.addWidget(self.video_widget)  # 1 = 影片

        # 進度條與控制列（影片）
        self.seek = QSlider(Qt.Orientation.Horizontal, self)
        self.seek.setEnabled(False)
        if _MULTIMEDIA_AVAILABLE:
            self.seek.sliderMoved.connect(self.player.setPosition)
            self.player.durationChanged.connect(
                lambda d: (self.seek.setRange(0, d), self.seek.setEnabled(True))
            )
            self.player.positionChanged.connect(self.seek.setValue)

        # 控制列包在容器，方便整體顯示/隱藏
        self.ctrl_bar = QW(self)
        ctrl_row = QHBoxLayout(self.ctrl_bar)
        ctrl_row.setContentsMargins(0, 0, 0, 0)
        self.btn_play = QPushButton("播放/暫停", self.ctrl_bar)
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_set_in = QPushButton("設為入點", self.ctrl_bar)
        self.btn_set_out = QPushButton("設為出點", self.ctrl_bar)
        self.btn_trim = QPushButton("剪出新檔", self.ctrl_bar)

        self.btn_mute = QPushButton("去音另存", self.ctrl_bar)
        for b in (
            self.btn_set_in,
            self.btn_set_out,
            self.btn_trim,
            self.btn_mute,
            self.btn_play,
        ):
            b.setStyleSheet(_BTN_STYLE)
        if _MULTIMEDIA_AVAILABLE:
            self.btn_set_in.clicked.connect(self._mark_in)
            self.btn_set_out.clicked.connect(self._mark_out)
            self.btn_trim.clicked.connect(self._ffmpeg_trim)
            self.btn_mute.clicked.connect(self._ffmpeg_mute)
        # 預覽影格不依賴多媒體模組，皆可使用（以 ffmpeg 擷取影格）

        ctrl_row.addWidget(self.btn_play)
        ctrl_row.addWidget(self.btn_set_in)
        ctrl_row.addWidget(self.btn_set_out)
        ctrl_row.addWidget(self.btn_trim)

        ctrl_row.addWidget(self.btn_mute)

        # 左側容器
        left_panel = QW(self)
        lpv = QVBoxLayout(left_panel)
        lpv.setContentsMargins(0, 0, 0, 0)
        lpv.addLayout(self.left_stack, 1)
        lpv.addWidget(self.seek)
        lpv.addWidget(self.ctrl_bar)

        # 右側：拖放 + 檔案操作
        self.drop = QLabel("拖放檔案到此處，或點「選擇檔案」", self)
        self.drop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop.setFixedHeight(120)
        self.drop.setStyleSheet(
            """
            QLabel { border: 2px dashed #555; border-radius: 12px; color: #AAA; font-size: 14px; background-color: #1b1b1b; font-family: "Source Han Sans TC"; }
            """
        )

        btns = QHBoxLayout()
        pick = QPushButton("選擇檔案", self)
        pick.clicked.connect(self.pick_file)
        clear = QPushButton("清除", self)
        clear.clicked.connect(self.clear_preview)
        self.save = QPushButton("儲存", self)
        self.save.clicked.connect(self._save_with_prompt)
        self.enqueue_btn = QPushButton("加入佇列", self)
        # OpenSpec: add-processing-queue — enqueue UI
        # spec: openspec/changes/add-processing-queue/specs/processing-queue/spec.md:7
        self.enqueue_btn.clicked.connect(self._enqueue_current)
        btns.addStretch()
        btns.addWidget(pick)
        btns.addWidget(clear)
        btns.addWidget(self.enqueue_btn)
        pick.setStyleSheet(_BTN_STYLE)
        clear.setStyleSheet(_BTN_STYLE)
        self.save.setStyleSheet(_SAVE_STYLE)
        self.enqueue_btn.setStyleSheet(_BTN_STYLE)
        self.save.setDisabled(True)

        right_panel = QW(self)
        rv = QVBoxLayout(right_panel)
        rv.setContentsMargins(16, 0, 16, 0)
        rv.setSpacing(12)
        rv.addWidget(self.drop)
        rv.addLayout(btns)
        self.rem_bg = QCheckBox()
        self.rem_bg.setText("remove background")
        self.rem_bg.setStyleSheet("font-weight: 300; font-size: 14px; color: #FFFFFF;")
        rv.addWidget(self.rem_bg)

        # 引擎選擇
        eng_row = QHBoxLayout()
        eng_row.addWidget(QLabel("引擎:", self))
        self.engine_box = QComboBox(self)
        self.engine_box.addItems(["hsv", "rembg", "openvino", "wand"])
        self.engine_box.currentTextChanged.connect(self._on_engine_changed)
        eng_row.addWidget(self.engine_box)
        rv.addLayout(eng_row)

        # HSV 控制（僅 engine=hsv 時啟用）
        hsv_row1 = QHBoxLayout()
        hsv_row1.addWidget(QLabel("H", self))
        self.s_h = QSlider(Qt.Orientation.Horizontal)
        self.s_h.setRange(1, 60)
        self.s_h.setValue(10)
        hsv_row1.addWidget(self.s_h)
        hsv_row1.addWidget(QLabel("S", self))
        self.s_s = QSlider(Qt.Orientation.Horizontal)
        self.s_s.setRange(1, 100)
        self.s_s.setValue(60)
        hsv_row1.addWidget(self.s_s)
        hsv_row1.addWidget(QLabel("V", self))
        self.s_v = QSlider(Qt.Orientation.Horizontal)
        self.s_v.setRange(1, 100)
        self.s_v.setValue(60)
        hsv_row1.addWidget(self.s_v)
        rv.addLayout(hsv_row1)

        hsv_row2 = QHBoxLayout()
        hsv_row2.addWidget(QLabel("倍率", self))
        self.s_strength = QSlider(Qt.Orientation.Horizontal)
        self.s_strength.setRange(50, 300)
        self.s_strength.setValue(150)  # 0.5~3.0
        hsv_row2.addWidget(self.s_strength)
        hsv_row2.addWidget(QLabel("侵蝕", self))
        self.s_erode = QSlider(Qt.Orientation.Horizontal)
        self.s_erode.setRange(0, 5)
        self.s_erode.setValue(1)
        hsv_row2.addWidget(self.s_erode)
        hsv_row2.addWidget(QLabel("膨脹", self))
        self.s_dilate = QSlider(Qt.Orientation.Horizontal)
        self.s_dilate.setRange(0, 5)
        self.s_dilate.setValue(0)
        hsv_row2.addWidget(self.s_dilate)
        rv.addLayout(hsv_row2)

        hsv_row3 = QHBoxLayout()
        hsv_row3.addWidget(QLabel("羽化", self))
        self.s_feather = QSlider(Qt.Orientation.Horizontal)
        self.s_feather.setRange(0, 20)
        self.s_feather.setValue(2)
        hsv_row3.addWidget(self.s_feather)
        self.cb_guided = QCheckBox("導向濾波")
        self.cb_guided.setStyleSheet("color:#FFFFFF")
        hsv_row3.addWidget(self.cb_guided)
        rv.addLayout(hsv_row3)

        prev_row = QHBoxLayout()

        self.btn_preview = QPushButton("預覽")
        self.btn_preview.setStyleSheet(_BTN_STYLE)
        self.btn_preview.clicked.connect(self._on_preview_clicked)

        prev_row.addStretch(1)
        prev_row.addWidget(self.btn_preview)
        rv.addLayout(prev_row)

        # Wand 控件（engine=wand 顯示）
        wand_row = QHBoxLayout()
        wand_row.addWidget(QLabel("容差(H)", self))
        self.wand_tol = QSlider(Qt.Orientation.Horizontal)
        self.wand_tol.setRange(1, 60)
        self.wand_tol.setValue(10)
        wand_row.addWidget(self.wand_tol)
        self.btn_wand = QPushButton("選取區域")
        self.btn_wand.setStyleSheet(_BTN_STYLE)
        self.btn_wand.clicked.connect(self._open_wand_dialog)
        wand_row.addWidget(self.btn_wand)
        rv.addLayout(wand_row)
        # 簡易佇列狀態列
        qrow = QHBoxLayout()
        self.queue_label = QLabel("Queue: 0 pending", self)
        self.q_pause = QPushButton("暫停")
        self.q_resume = QPushButton("繼續")
        self.q_cancel = QPushButton("取消當前")
        for b in (self.q_pause, self.q_resume, self.q_cancel):
            b.setStyleSheet(_BTN_STYLE)
        self.q_pause.clicked.connect(lambda: self._queue_cmd("pause"))
        self.q_resume.clicked.connect(lambda: self._queue_cmd("resume"))
        self.q_cancel.clicked.connect(lambda: self._queue_cmd("cancel"))
        qrow.addWidget(self.queue_label)
        qrow.addStretch(1)
        qrow.addWidget(self.q_pause)
        qrow.addWidget(self.q_resume)
        qrow.addWidget(self.q_cancel)
        rv.addLayout(qrow)
        rv.addStretch(1)
        rv.addWidget(self.save)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        root.addWidget(title)
        root.addWidget(splitter, 1)

        # 剪輯狀態
        self._in_ms = None
        self._out_ms = None

        # 預設為圖片模式：隱藏影片控制列
        self._set_controls_visible(False)

        # 讓預覽接收點擊（魔術棒）
        self.preview_img.installEventFilter(self)
        self._last_img_size = None  # (w,h)
        self._orig_pix = None
        # 以設定檔預設初始化 HSV 控件
        try:
            p = self.parent()
            if p and hasattr(p, "config"):
                hcfg = p.config.get("hsv", {})
                self.s_h.setValue(int(hcfg.get("tol_h", 10)))
                self.s_s.setValue(int(hcfg.get("tol_s", 60)))
                self.s_v.setValue(int(hcfg.get("tol_v", 60)))
                self.s_strength.setValue(int(float(hcfg.get("strength", 1.5)) * 100))
                self.s_erode.setValue(int(hcfg.get("erode_iter", 1)))
                self.s_dilate.setValue(int(hcfg.get("dilate_iter", 0)))
                self.s_feather.setValue(int(float(hcfg.get("feather_px", 2.0))))
                self.cb_guided.setChecked(bool(hcfg.get("use_guided", False)))
        except Exception:
            pass
        # 即時預覽事件綁定
        for s in (
            self.s_h,
            self.s_s,
            self.s_v,
            self.s_strength,
            self.s_erode,
            self.s_dilate,
            self.s_feather,
            self.wand_tol,
        ):
            s.valueChanged.connect(self._update_preview_if_live)
        if hasattr(self, "cb_guided"):
            self.cb_guided.toggled.connect(self._update_preview_if_live)
        # 佇列統計（由 Main 綁定更新）
        self._queue_pending = 0

    # ---------- 顯示/隱藏影片控制 ----------
    def _set_controls_visible(self, visible: bool):
        self.seek.setVisible(visible and _MULTIMEDIA_AVAILABLE)
        self.ctrl_bar.setVisible(visible and _MULTIMEDIA_AVAILABLE)
        # self.rem_bg.setVisible(not visible)

    # ---------- file picking ----------
    def pick_file(self):
        dlg = QFileDialog(self, "選擇媒體")
        dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
        mimes = ["image/gif", "image/webp", "image/png", "image/jpeg"]
        if _MULTIMEDIA_AVAILABLE:
            mimes += [
                "video/mp4",
                "video/x-matroska",
                "video/quicktime",
                "video/x-msvideo",
                "video/webm",
            ]
        dlg.setMimeTypeFilters(mimes)
        if dlg.exec():
            path = dlg.selectedFiles()[0]
            self.load_path(path)
            self.save.setEnabled(True)

    # ---------- drag & drop ----------
    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            for u in e.mimeData().urls():
                if u.isLocalFile() and self._is_allowed(u.toLocalFile()):
                    e.acceptProposedAction()
                    return
        e.ignore()

    def dropEvent(self, e: QDropEvent):
        for u in e.mimeData().urls():
            if u.isLocalFile():
                path = u.toLocalFile()
                if self._is_allowed(path):
                    self.load_path(path)
                    e.acceptProposedAction()
                    self.save.setEnabled(True)
                    return
        e.ignore()

    # ---------- core ----------
    def load_path(self, path: str):
        try:
            self._current_path = path
            self.fileSelected.emit(path)
            ext = os.path.splitext(path)[1].lower()

            if ext in VIDEO_EXTS:
                if _MULTIMEDIA_AVAILABLE:
                    # 影片模式
                    self._clear_movie()
                    self.player.setSource(QUrl.fromLocalFile(path))
                    self.left_stack.setCurrentIndex(1)
                    self.seek.setEnabled(False)
                    self._set_controls_visible(True)
                    self.player.play()
                    return
                else:
                    # 在缺多媒體相依時，提示無法播放影片
                    self.toast.emit(
                        {
                            "level": "warn",
                            "title": "缺少多媒體相依",
                            "message": "影片播放需要系統安裝 PulseAudio / GStreamer（Linux/WSL）。已關閉影片功能。",
                            "duration": 5000,
                        }
                    )
                    # 直接下方嘗試當作圖片會失敗，交由靜態圖流程處理（顯示空）

            # 圖片/GIF 模式
            reader = QImageReader(path)
            if reader.supportsAnimation():
                movie = QMovie(path)
                if movie.isValid():
                    self.left_stack.setCurrentIndex(0)
                    self._set_controls_visible(False)  # GIF 隱藏下方按鈕
                    self._set_movie(movie)
                    return
            self._clear_movie()
            self.clear_preview()
            pix = QPixmap(path)
            if not pix.isNull():
                self.left_stack.setCurrentIndex(0)
                self._set_controls_visible(False)  # 靜態圖也隱藏
                # 保留原圖 pixmap 供預覽疊色使用
                self._orig_pix = QPixmap(pix)
                self.preview_img.setPixmap(self._scaled(self._orig_pix))
                try:
                    self._preview_service.set_static_source(path)
                    self._preview_source_path = path
                except Exception:
                    self._preview_source_path = None
                # 記錄原始尺寸供座標轉換
                sz = reader.size()
                self._last_img_size = (sz.width(), sz.height())
        except Exception as e:
            self.on_exception.emit(e)

    def clear_preview(self):
        self._current_path = None
        self._clear_movie()
        self.preview_img.clear()
        self.preview_img.setText("無預覽")
        if _MULTIMEDIA_AVAILABLE:
            self.player.stop()
        self.seek.setEnabled(False)
        self._set_controls_visible(False)
        self.save.setDisabled(True)
        # 清除原圖快取
        if hasattr(self, "_orig_pix"):
            self._orig_pix = None
        self._preview_source_path = None
        try:
            self._preview_service.clear_cache()
        except Exception:
            pass

    # ---------- helpers ----------
    def _is_allowed(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in {".gif", ".webp", ".png", ".jpg", ".jpeg", ".apng", *VIDEO_EXTS}

    def _set_movie(self, movie: QMovie):
        self._clear_movie()
        self._movie = movie
        self.preview_img.setMovie(self._movie)
        self._movie.setScaledSize(self._fit_size(self.preview_img.size()))
        self._movie.start()

    def _clear_movie(self):
        if self._movie:
            self._movie.stop()
            self._movie.deleteLater()
            self._movie = None

    def _save_with_prompt(self):
        try:
            if not self._current_path:
                return
            dialog_result = self.collect_save_input()
            if not dialog_result:
                return
            self.persist_output_settings(dialog_result.options)
            plan = self.build_save_plan(dialog_result)
            if not plan:
                return
            self.run_save_plan(plan)
        except Exception as e:
            self.on_exception.emit(e)

    def collect_save_input(self) -> Optional[SaveDialogResult]:
        if not self._current_path:
            return None
        from ui.save_dialog import SaveDialog

        cfg = getattr(self.p, "config", {}) if hasattr(self, "p") else {}
        defaults = merge_export_settings(cfg if isinstance(cfg, dict) else {}, None)
        orig_name = os.path.basename(self._current_path)
        base, _ = os.path.splitext(orig_name)
        default_ext = ".webp"
        default_name = base + default_ext
        dlg = SaveDialog(
            self,
            default_name=default_name,
            quality=defaults.quality,
            max_fps=(defaults.max_fps if defaults.max_fps > 0 else 15),
            auto_fps=(defaults.max_fps <= 0),
            loop=defaults.loop,
            enable_size=False,
        )
        if dlg.exec() != 1:
            return None
        payload = dlg.result_payload()
        name = (payload.get("name") or "").strip()
        if not name:
            return None
        ext = os.path.splitext(name)[1].lower()
        if ext == "":
            name += default_ext
            ext = default_ext
        profile = ext.lstrip(".") or "webp"

        options = ExportOptions(
            quality=int(payload.get("quality", defaults.quality)),
            max_fps=0
            if bool(payload.get("max_fps_auto", False))
            else int(
                payload.get("max_fps", defaults.max_fps if defaults.max_fps > 0 else 15)
            ),
            loop=bool(payload.get("loop", defaults.loop)),
            profile=profile,
        )
        self._last_export_options = options
        self.toast.emit(
            {
                "level": "info",
                "title": "輸出格式",
                "message": f"輸出將以 {profile.upper()} 儲存（動圖為 animated WebP）。",
                "duration": 2500,
            }
        )
        return SaveDialogResult(filename=name, options=options)

    def persist_output_settings(self, options: ExportOptions) -> None:
        self._save_controller.persist_output_settings(options)

    def build_save_plan(self, result: SaveDialogResult) -> Optional[SavePlan]:
        return self._save_controller.build_save_plan(
            self._current_path, result, self.rem_bg.isChecked()
        )

    def run_save_plan(self, plan: SavePlan) -> None:
        self._save_controller.run_save_plan(plan)

    def _enqueue_job_request(self, payload: JobRequest) -> None:
        self.enqueueJob.emit(*payload.to_payload())

    def _set_last_export_options(self, options: ExportOptions) -> None:
        self._last_export_options = options

    def _show_message(self, title: str, text: str) -> None:
        dlg = QMessageBox(self)
        dlg.setWindowTitle(title)
        dlg.setText(text)
        dlg.setIcon(QMessageBox.Icon.Information)
        try:
            dlg.setStyleSheet(_BASE)
        except Exception:
            pass
        dlg.exec()

    def _confirm_overwrite(self, path: str) -> bool:
        if not os.path.exists(path):
            return True
        dlg = QMessageBox(self)
        dlg.setWindowTitle("覆寫確認")
        dlg.setText(f"檔案「{os.path.basename(path)}」已存在，是否覆寫？")
        dlg.setIcon(QMessageBox.Icon.Question)
        yes = dlg.addButton("是", QMessageBox.ButtonRole.YesRole)
        dlg.addButton("否", QMessageBox.ButtonRole.NoRole)
        try:
            dlg.setStyleSheet(_BASE)
        except Exception:
            pass
        dlg.exec()
        return dlg.clickedButton() is yes

    def _build_queue_job(
        self,
        src_path: str,
        destination: str,
        options: ExportOptions,
        prefer_override: Optional[str],
        rem_bg_enabled: bool,
    ) -> Optional[JobRequest]:
        if not src_path:
            return None
        diag_id = generate_diagnostic_id()
        if rem_bg_enabled and prefer_override is None:
            prefer = (
                getattr(self, "engine_box", None).currentText()
                if hasattr(self, "engine_box")
                else "hsv"
            )
            opts = self._collect_engine_options(prefer)
            if opts is None:
                return None
            range_ms = self._collect_range()
            if range_ms:
                opts.setdefault("range", {})
                if range_ms[0] is not None:
                    opts["range"]["in_ms"] = range_ms[0]
                if range_ms[1] is not None:
                    opts["range"]["out_ms"] = range_ms[1]
            opts.setdefault("export", options.to_dict())
            opts.setdefault("diagnostic_id", diag_id)
            temp_path = self._temp_manager.copy_to_temp(
                src_path,
                preferred_name=os.path.basename(destination),
                persistent=True,
                diagnostic_id=diag_id,
            )
            log_structured(
                self.logger,
                logging.INFO,
                diag_id,
                "queue.job.prepared",
                src=temp_path,
                prefer=prefer or "auto",
                destination=destination,
            )
            return JobRequest(
                src=temp_path, prefer=prefer or "auto", opts=opts, diagnostic_id=diag_id
            )

        prefer = prefer_override or "none"
        opts = {"export": options.to_dict(), "diagnostic_id": diag_id}
        range_ms = self._collect_range()
        if range_ms:
            opts.setdefault("range", {})
            if range_ms[0] is not None:
                opts["range"]["in_ms"] = range_ms[0]
            if range_ms[1] is not None:
                opts["range"]["out_ms"] = range_ms[1]
        temp_path = self._temp_manager.copy_to_temp(
            src_path,
            preferred_name=os.path.basename(destination),
            persistent=True,
            diagnostic_id=diag_id,
        )
        log_structured(
            self.logger,
            logging.INFO,
            diag_id,
            "queue.job.prepared",
            src=temp_path,
            prefer=prefer,
            destination=destination,
        )
        return JobRequest(src=temp_path, prefer=prefer, opts=opts, diagnostic_id=diag_id)

    def _collect_engine_options(self, prefer: str) -> Optional[Dict[str, Any]]:
        prefer = prefer or "hsv"
        if prefer == "wand":
            seed = getattr(self, "_wand_seed", None)
            if seed is None:
                self.toast.emit(
                    {
                        "level": "warn",
                        "title": "魔術棒",
                        "message": "請先選取區域",
                        "duration": 3000,
                    }
                )
                return None
            return {
                "seed": seed,
                "tolH": int(self.wand_tol.value()),
                "tolS": 60,
                "tolV": 60,
                "contiguous": True,
            }
        if prefer == "hsv":
            return {
                "hsv": {
                    "tol_h": int(self.s_h.value()),
                    "tol_s": int(self.s_s.value()),
                    "tol_v": int(self.s_v.value()),
                }
            }
        return {}

    def _collect_range(self) -> Optional[Tuple[Optional[int], Optional[int]]]:
        if self._in_ms is None and self._out_ms is None:
            return None
        try:
            start = int(self._in_ms) if self._in_ms is not None else None
            end = int(self._out_ms) if self._out_ms is not None else None
            return start, end
        except Exception:
            return None

    def _can_direct_copy(self, src_path: str, destination: str) -> bool:
        if not src_path:
            return False
        src_ext = os.path.splitext(src_path)[1].lower()
        dest_ext = os.path.splitext(destination)[1].lower()
        return src_ext == dest_ext and dest_ext in _EXTS

    def _remove_bg(self):
        pass

    # ========== 影片工具 ==========

    def eventFilter(self, obj, ev):
        # 魔術棒：在圖片模式、engine=wand 下，記錄點擊座標
        if obj is self.preview_img and ev.type() == QEvent.Type.MouseButtonPress:
            if (
                hasattr(self, "engine_box")
                and self.engine_box.currentText() == "wand"
                and self._current_path
                and self.left_stack.currentIndex() == 0
            ):
                pos = ev.position().toPoint()
                mapped = self._map_click_to_image(pos)
                if mapped is not None:
                    self._wand_seed = mapped  # (x,y)
                    self.toast.emit(
                        {
                            "level": "info",
                            "title": "魔術棒",
                            "message": f"取樣座標：{mapped}",
                            "duration": 1500,
                        }
                    )
        return super().eventFilter(obj, ev)

    def _map_click_to_image(self, pt):
        if self._last_img_size is None:
            return None
        label_size = self.preview_img.size()
        view_w = max(1, label_size.width() - 16)
        view_h = max(1, label_size.height() - 16)
        src_w, src_h = self._last_img_size
        k = min(view_w / src_w, view_h / src_h)
        disp_w, disp_h = int(src_w * k), int(src_h * k)
        # 圖像顯示區居中，計算偏移
        off_x = (label_size.width() - disp_w) // 2
        off_y = (label_size.height() - disp_h) // 2
        x = pt.x() - off_x
        y = pt.y() - off_y
        if x < 0 or y < 0 or x >= disp_w or y >= disp_h:
            return None
        # 映回原圖座標
        src_x = int(x / k)
        src_y = int(y / k)
        return (src_x, src_y)

    def _open_wand_dialog(self):
        # 圖片模式：直接用當前路徑；影片模式：擷取當前影格
        if not self._current_path:
            return
        path = self._current_path
        if _MULTIMEDIA_AVAILABLE and self.left_stack.currentIndex() == 1:
            tmp = self._extract_current_frame()
            if tmp:
                path = tmp
        from ui.wand_dialog import WandDialog

        dlg = WandDialog(path, tol_h=self.wand_tol.value(), parent=self)
        res = dlg.exec()
        out = dlg.get_result()
        if res and out:
            seed, tol_h = out
            self._wand_seed = seed
            self.wand_tol.setValue(int(tol_h))
            # 預覽
            self._update_preview()

    def _extract_current_frame(self) -> str | None:
        try:
            if not self._current_path:
                return None
            os.makedirs("./temp", exist_ok=True)
            out = os.path.join("./temp", "preview_frame.png")
            t_ms = 0
            if _MULTIMEDIA_AVAILABLE and hasattr(self, "player"):
                t_ms = int(self.player.position())
            args = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{t_ms / 1000:.3f}",
                "-i",
                self._current_path,
                "-frames:v",
                "1",
                out,
            ]
            args = self._command_builder.apply_profile(
                args, ExportCommandBuilder.PROFILE_WEBP, self._last_export_options
            )
            ok, _ = self._ffmpeg(args)
            return out if ok else None
        except Exception:
            return None

    # def _open_video_preview(self):
    #     try:
    #         path = self._extract_current_frame()
    #         if not path:
    #             self.toast.emit({"level":"error","title":"預覽失敗","message":"擷取影格失敗","duration":3000})
    #             return
    #         from ui.preview_dialog import PreviewImageDialog
    #         init_hsv = {
    #             "tol_h": int(self.s_h.value()),
    #             "tol_s": int(self.s_s.value()),
    #             "tol_v": int(self.s_v.value()),
    #             "strength": float(self.s_strength.value())/100.0,
    #         }
    #         dlg = PreviewImageDialog(path, refresh_fn=self._extract_current_frame, init_hsv=init_hsv, parent=self)
    #         # 接收預覽對話框回傳，套用到本頁控制
    #         dlg.hsvChanged.connect(self._apply_hsv_from_preview)
    #         dlg.seedSelected.connect(self._apply_seed_from_preview)
    #         dlg.exec()
    #     except Exception as e:
    #         self.on_exception.emit(e)

    def _open_video_preview(self):
        try:
            self.toast.emit(
                {
                    "level": "info",
                    "title": "預覽",
                    "message": "正在開啟預覽視窗…",
                    "duration": 1500,
                }
            )
            from ui.preview_dialog import PreviewImageDialog
            from ui.services.preview_service import PreviewService

            if not getattr(self, "_current_path", None):
                self.toast.emit(
                    {
                        "level": "error",
                        "title": "錯誤",
                        "message": "請先載入影片",
                        "duration": 3000,
                    }
                )
                return
            init_hsv = {
                "tol_h": int(self.s_h.value()),
                "tol_s": int(self.s_s.value()),
                "tol_v": int(self.s_v.value()),
                "strength": float(self.s_strength.value()) / 100.0,
            }
            init_ms = None
            try:
                if globals().get("_MULTIMEDIA_AVAILABLE") and hasattr(self, "player"):
                    init_ms = int(self.player.position())
            except Exception:
                init_ms = None
            service = PreviewService()
            dlg = PreviewImageDialog(
                self._current_path,
                refresh_fn=None,
                init_hsv=init_hsv,
                init_ms=init_ms,
                export_options=self._last_export_options,
                service=service,
                parent=self,
            )
            dlg.hsvChanged.connect(self._apply_hsv_from_preview)
            dlg.seedSelected.connect(self._apply_seed_from_preview)
            dlg.exec()
        except Exception as e:
            self.on_exception.emit(e)

    def _apply_hsv_from_preview(self, payload: dict):
        try:
            if "tol_h" in payload:
                self.s_h.setValue(int(payload["tol_h"]))
            if "tol_s" in payload:
                self.s_s.setValue(int(payload["tol_s"]))
            if "tol_v" in payload:
                self.s_v.setValue(int(payload["tol_v"]))
            if "strength" in payload:
                self.s_strength.setValue(int(float(payload["strength"]) * 100))
            self._update_preview_if_live()
        except Exception:
            pass

    def _apply_seed_from_preview(self, seed):
        try:
            if isinstance(seed, (tuple, list)) and len(seed) == 2:
                self._wand_seed = (int(seed[0]), int(seed[1]))
                self.toast.emit(
                    {
                        "level": "info",
                        "title": "魔術棒",
                        "message": f"取樣座標：{self._wand_seed}",
                        "duration": 1500,
                    }
                )
        except Exception:
            pass

    # ---------- Queue helpers ----------
    def _enqueue_current(self):
        try:
            if not self._current_path:
                self.toast.emit(
                    {
                        "level": "warn",
                        "title": "加入佇列",
                        "message": "請先選擇檔案",
                        "duration": 2000,
                    }
                )
                return
            prefer = (
                getattr(self, "engine_box", None).currentText()
                if hasattr(self, "engine_box")
                else "hsv"
            )
            opts: dict = {}
            # 也在「加入佇列」按鈕加入 in/out
            if self._in_ms is not None or self._out_ms is not None:
                try:
                    opts.setdefault("range", {})
                    if self._in_ms is not None:
                        opts["range"]["in_ms"] = int(self._in_ms)
                    if self._out_ms is not None:
                        opts["range"]["out_ms"] = int(self._out_ms)
                except Exception:
                    pass
            if prefer == "wand":
                seed = getattr(self, "_wand_seed", None)
                if seed is None:
                    self.toast.emit(
                        {
                            "level": "warn",
                            "title": "魔術棒",
                            "message": "請先選取區域",
                            "duration": 2000,
                        }
                    )
                    return
                opts = {
                    "seed": seed,
                    "tolH": int(self.wand_tol.value()),
                    "tolS": 60,
                    "tolV": 60,
                    "contiguous": True,
                }
            elif prefer == "hsv":
                opts = {
                    "hsv": {
                        "tol_h": int(self.s_h.value()),
                        "tol_s": int(self.s_s.value()),
                        "tol_v": int(self.s_v.value()),
                    }
                }
            # 將臨時檔交由主程式產生後執行（沿用 _save_with_prompt 的邏輯）
            # 這裡直接用來源檔路徑，Main 會在實際執行時處理 temp 與輸出
            if isinstance(opts, dict) and hasattr(self, "_last_export_options"):
                opts.setdefault("export", self._last_export_options.to_dict())
            self.enqueueJob.emit(self._current_path, prefer, opts)
            self._queue_pending += 1
            self.queue_label.setText(f"Queue: {self._queue_pending} pending")
        except Exception as e:
            self.on_exception.emit(e)

    def _queue_cmd(self, cmd: str):
        # 交由 Main 透過 ProcessingQueue 實作；這裡只傳遞 toast 提示
        try:
            p = self.p
            if not p or not hasattr(p, "queue"):
                p.logger.error("無法存取處理佇列")
                return
            q = getattr(p, "queue")
            if cmd == "pause":
                q.pause()
                self.toast.emit(
                    {
                        "level": "info",
                        "title": "佇列",
                        "message": "已暫停",
                        "duration": 1500,
                    }
                )
            elif cmd == "resume":
                q.resume()
                self.toast.emit(
                    {
                        "level": "info",
                        "title": "佇列",
                        "message": "繼續",
                        "duration": 1500,
                    }
                )
            elif cmd == "cancel":
                q.cancel_current()
                if hasattr(p, "rmbg_thread"):
                    p.rmbg_thread.cancel_current()
                self.toast.emit(
                    {
                        "level": "warn",
                        "title": "佇列",
                        "message": "已取消當前（若可）",
                        "duration": 1500,
                    }
                )
        except Exception as e:
            self.on_exception.emit(e)

    def _on_engine_changed(self, eng: str):
        is_hsv = eng == "hsv"
        is_wand = eng == "wand"
        for w in (
            self.s_h,
            self.s_s,
            self.s_v,
            self.s_strength,
            self.s_erode,
            self.s_dilate,
            self.s_feather,
            self.cb_guided,
        ):
            w.setEnabled(is_hsv)
        self.btn_wand.setEnabled(is_wand)
        self.wand_tol.setEnabled(is_wand)

    def _update_preview(self):
        try:
            if not self._current_path or self.left_stack.currentIndex() != 0:
                return
            eng = (
                self.engine_box.currentText() if hasattr(self, "engine_box") else "hsv"
            )
            seed = None
            if eng not in ("hsv", "wand"):
                self.toast.emit(
                    {
                        "level": "info",
                        "title": "預覽",
                        "message": "此引擎不支援即時預覽",
                        "duration": 2000,
                    }
                )
                return

            if self._preview_source_path != self._current_path:
                self._preview_service.set_static_source(self._current_path)
                self._preview_source_path = self._current_path

            if eng == "wand":
                seed = getattr(self, "_wand_seed", None)
                if not seed:
                    self.toast.emit(
                        {
                            "level": "warn",
                            "title": "魔術棒",
                            "message": "請先選取區域",
                            "duration": 2000,
                        }
                    )
                    return
                self._preview_service.update_seed(seed)
                wand_opts = {
                    "tolH": int(self.wand_tol.value()),
                    "tolS": 60,
                    "tolV": 60,
                    "contiguous": True,
                    "use_edge_barrier": True,
                    "connectivity": 8,
                }
            else:
                self._preview_service.update_seed(None)
                wand_opts = None

            self._preview_service.update_hsv("tol_h", int(self.s_h.value()))
            self._preview_service.update_hsv("tol_s", int(self.s_s.value()))
            self._preview_service.update_hsv("tol_v", int(self.s_v.value()))
            self._preview_service.update_hsv(
                "strength", float(self.s_strength.value()) / 100.0
            )
            self._preview_service.update_hsv("erode_iter", int(self.s_erode.value()))
            self._preview_service.update_hsv("dilate_iter", int(self.s_dilate.value()))
            self._preview_service.update_hsv("feather_px", float(self.s_feather.value()))
            self._preview_service.update_hsv(
                "use_guided", float(bool(self.cb_guided.isChecked()))
            )

            frame = self._preview_service.load_current_frame_with_engine(
                engine=eng,
                wand_seed=seed if eng == "wand" else None,
                wand_opts=wand_opts,
                apply_overlay=True,
            )
            qov = frame.overlay
            if qov is None:
                return

            from PyQt6.QtGui import QPainter

            # 以原圖為底，避免覆疊多次後失真或透明
            base = getattr(self, "_orig_pix", None)
            if base is None or base.isNull():
                base = QPixmap.fromImage(frame.image)
                if base.isNull():
                    return
                self._orig_pix = QPixmap(base)
            scaled = self._scaled(base)
            qov = qov.scaled(
                scaled.size(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            out = QPixmap(scaled.size())
            out.fill(Qt.GlobalColor.transparent)
            p = QPainter(out)
            p.drawPixmap(0, 0, scaled)
            p.drawImage(0, 0, qov)
            p.end()
            self.preview_img.setPixmap(out)
        except Exception as e:
            self.on_exception.emit(e)

    def _update_preview_if_live(self, *args, **kwargs):
        try:
            if hasattr(self, "cb_live") and self.cb_live.isChecked():
                self._update_preview()
        except Exception as e:
            self.on_exception.emit(e)

    def _get_out_dir(self) -> str:
        # 從父視窗（Main）讀取設定輸出資料夾
        try:
            parent = self.parent()
            if parent and hasattr(parent, "config"):
                od = parent.config.get("output", {}).get("dir", "./animes")
                return od or "./animes"
        except Exception:
            pass
        return os.path.join(os.getcwd(), "animes")

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._movie:
            self._movie.setScaledSize(self._fit_size(self.preview_img.size()))
        elif self.preview_img.pixmap() is not None:
            self.preview_img.setPixmap(self._scaled(self.preview_img.pixmap()))

    def _fit_size(self, box: QSize) -> QSize:
        return QSize(max(1, box.width() - 16), max(1, box.height() - 16))

    def _scaled(self, pix: QPixmap) -> QPixmap:
        return pix.scaled(
            self.preview_img.size() - QSize(16, 16),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    # ---------- Video controls ----------
    def _toggle_play(self):
        st = self.player.playbackState()
        if st == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _ask_output(self, suffix: str):
        base = self._video_service.suggest_output_name(
            self._current_path or "output", suffix
        )
        text, ok = QInputDialog.getText(
            self, "輸出檔名", "輸入檔名：", text=base
        )
        if not ok or not text.strip():
            return None
        out_dir = os.path.join(os.getcwd(), "animes")
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, text.strip())

    def _run_ffmpeg_command(self, args: list[str]) -> Tuple[bool, str]:
        try:
            proc = subprocess.run(
                args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
            return True, proc.stderr.decode("utf-8", "ignore")
        except subprocess.CalledProcessError as e:
            return False, e.stderr.decode("utf-8", "ignore")

    def _ffmpeg(self, args: list[str]):
        ok, log = self._run_ffmpeg_command(args)
        if not ok:
            self.on_exception.emit(Exception(log))
        return ok, log

    def _mark_in(self):
        if self.left_stack.currentIndex() == 1:
            self._in_ms = self.player.position()
            self.toast.emit(
                {
                    "level": "info",
                    "title": "入點",
                    "message": f"{self._in_ms / 1000:.2f}s",
                    "duration": 2000,
                }
            )

    def _mark_out(self):
        if self.left_stack.currentIndex() == 1:
            self._out_ms = self.player.position()
            self.toast.emit(
                {
                    "level": "info",
                    "title": "出點",
                    "message": f"{self._out_ms / 1000:.2f}s",
                    "duration": 2000,
                }
            )

    def _ffmpeg_trim(self):
        if (
            not self._current_path
            or self._in_ms is None
            or self._out_ms is None
            or self._out_ms <= self._in_ms
        ):
            return
        out = self._ask_output("_trim.mp4")
        if not out:
            return
        diag_id = generate_diagnostic_id()
        log_structured(
            self.logger,
            logging.INFO,
            diag_id,
            "video.trim.request",
            src=self._current_path,
            dest=out,
            in_ms=int(self._in_ms),
            out_ms=int(self._out_ms),
        )
        result = self._video_service.trim(
            self._current_path,
            int(self._in_ms),
            int(self._out_ms),
            out,
            self._last_export_options,
            diagnostic_id=diag_id,
        )
        if result.success:
            log_structured(
                self.logger, logging.INFO, diag_id, "video.trim.success", dest=out
            )
            self.toast.emit(
                {
                    "level": "info",
                    "title": "完成",
                    "message": f"已輸出 {os.path.basename(out)}（追蹤 {diag_id}）",
                    "duration": 4000,
                }
            )
            self.update_data.emit()
        else:
            err = attach_diagnostic(Exception(result.log), diag_id)
            log_structured(
                self.logger,
                logging.ERROR,
                diag_id,
                "video.trim.error",
                error=result.log,
            )
            self.on_exception.emit(err)

    def _ffmpeg_mute(self):
        if not self._current_path:
            return
        out = self._ask_output("_mute.mp4")
        if not out:
            return
        diag_id = generate_diagnostic_id()
        log_structured(
            self.logger,
            logging.INFO,
            diag_id,
            "video.mute.request",
            src=self._current_path,
            dest=out,
        )
        result = self._video_service.mute(
            self._current_path,
            out,
            self._last_export_options,
            diagnostic_id=diag_id,
        )
        if result.success:
            log_structured(
                self.logger, logging.INFO, diag_id, "video.mute.success", dest=out
            )
            self.toast.emit(
                {
                    "level": "info",
                    "title": "完成",
                    "message": f"已輸出 {os.path.basename(out)}（追蹤 {diag_id}）",
                    "duration": 4000,
                }
            )
            self.update_data.emit()
        else:
            err = attach_diagnostic(Exception(result.log), diag_id)
            log_structured(
                self.logger,
                logging.ERROR,
                diag_id,
                "video.mute.error",
                error=result.log,
            )
            self.on_exception.emit(err)

    def _on_preview_clicked(self):
        try:
            if not getattr(self, "_current_path", None):
                self.toast.emit(
                    {
                        "level": "error",
                        "title": "預覽失敗",
                        "message": "沒有可預覽的來源",
                        "duration": 3000,
                    }
                )
                return
            ext = os.path.splitext(self._current_path)[1].lower()
            is_video = ext in VIDEO_EXTS
            is_anim = False
            try:
                r = QImageReader(self._current_path)
                r.setDecideFormatFromContent(True)
                is_anim = r.supportsAnimation()
            except Exception:
                is_anim = False
            if is_video or is_anim:
                self._open_video_preview()
            else:
                self._update_preview()
        except Exception as e:
            self.on_exception.emit(e)
