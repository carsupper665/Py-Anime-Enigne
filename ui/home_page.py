# ui/home_page.py
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QFileDialog, QSplitter,
    QInputDialog, QMessageBox, QSlider, QStackedLayout, QWidget as QW, QCheckBox,
    QComboBox
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
import os, shutil, subprocess

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

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("HomePage")
        self._movie: QMovie | None = None
        self._current_path: str | None = None
        self.setAcceptDrops(True)
        self.setStyleSheet(_BASE)

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
            self.left_stack.addWidget(self.video_widget) # 1 = 影片

        # 進度條與控制列（影片）
        self.seek = QSlider(Qt.Orientation.Horizontal, self)
        self.seek.setEnabled(False)
        if _MULTIMEDIA_AVAILABLE:
            self.seek.sliderMoved.connect(self.player.setPosition)
            self.player.durationChanged.connect(lambda d: (self.seek.setRange(0, d), self.seek.setEnabled(True)))
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
        for b in (self.btn_set_in, self.btn_set_out, self.btn_trim, self.btn_mute, self.btn_play):
            b.setStyleSheet(_BTN_STYLE)
        if _MULTIMEDIA_AVAILABLE:
            self.btn_set_in.clicked.connect(self._mark_in)
            self.btn_set_out.clicked.connect(self._mark_out)
            self.btn_trim.clicked.connect(self._ffmpeg_trim)
            self.btn_mute.clicked.connect(self._ffmpeg_mute)
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
        btns.addStretch(); btns.addWidget(pick); btns.addWidget(clear)
        pick.setStyleSheet(_BTN_STYLE); clear.setStyleSheet(_BTN_STYLE); self.save.setStyleSheet(_SAVE_STYLE)
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
        self.s_h = QSlider(Qt.Orientation.Horizontal); self.s_h.setRange(1, 60); self.s_h.setValue(10)
        hsv_row1.addWidget(self.s_h)
        hsv_row1.addWidget(QLabel("S", self))
        self.s_s = QSlider(Qt.Orientation.Horizontal); self.s_s.setRange(1, 100); self.s_s.setValue(60)
        hsv_row1.addWidget(self.s_s)
        hsv_row1.addWidget(QLabel("V", self))
        self.s_v = QSlider(Qt.Orientation.Horizontal); self.s_v.setRange(1, 100); self.s_v.setValue(60)
        hsv_row1.addWidget(self.s_v)
        rv.addLayout(hsv_row1)

        hsv_row2 = QHBoxLayout()
        hsv_row2.addWidget(QLabel("倍率", self))
        self.s_strength = QSlider(Qt.Orientation.Horizontal); self.s_strength.setRange(50, 300); self.s_strength.setValue(150)  # 0.5~3.0
        hsv_row2.addWidget(self.s_strength)
        hsv_row2.addWidget(QLabel("侵蝕", self))
        self.s_erode = QSlider(Qt.Orientation.Horizontal); self.s_erode.setRange(0, 5); self.s_erode.setValue(1)
        hsv_row2.addWidget(self.s_erode)
        hsv_row2.addWidget(QLabel("膨脹", self))
        self.s_dilate = QSlider(Qt.Orientation.Horizontal); self.s_dilate.setRange(0, 5); self.s_dilate.setValue(0)
        hsv_row2.addWidget(self.s_dilate)
        rv.addLayout(hsv_row2)

        hsv_row3 = QHBoxLayout()
        hsv_row3.addWidget(QLabel("羽化", self))
        self.s_feather = QSlider(Qt.Orientation.Horizontal); self.s_feather.setRange(0, 20); self.s_feather.setValue(2)
        hsv_row3.addWidget(self.s_feather)
        self.cb_guided = QCheckBox("導向濾波"); self.cb_guided.setStyleSheet("color:#FFFFFF")
        hsv_row3.addWidget(self.cb_guided)
        rv.addLayout(hsv_row3)

        prev_row = QHBoxLayout()
        self.cb_live = QCheckBox("即時預覽"); self.cb_live.setStyleSheet("color:#FFFFFF")
        self.btn_preview = QPushButton("預覽"); self.btn_preview.setStyleSheet(_BTN_STYLE)
        self.btn_preview.clicked.connect(self._update_preview)
        prev_row.addWidget(self.cb_live)
        prev_row.addStretch(1)
        prev_row.addWidget(self.btn_preview)
        rv.addLayout(prev_row)

        # Wand 控件（engine=wand 顯示）
        wand_row = QHBoxLayout()
        wand_row.addWidget(QLabel("容差(H)", self))
        self.wand_tol = QSlider(Qt.Orientation.Horizontal); self.wand_tol.setRange(1, 60); self.wand_tol.setValue(10)
        wand_row.addWidget(self.wand_tol)
        self.btn_wand = QPushButton("選取區域"); self.btn_wand.setStyleSheet(_BTN_STYLE)
        self.btn_wand.clicked.connect(self._open_wand_dialog)
        wand_row.addWidget(self.btn_wand)
        rv.addLayout(wand_row)
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
            if p and hasattr(p, 'config'):
                hcfg = p.config.get('hsv', {})
                self.s_h.setValue(int(hcfg.get('tol_h', 10)))
                self.s_s.setValue(int(hcfg.get('tol_s', 60)))
                self.s_v.setValue(int(hcfg.get('tol_v', 60)))
                self.s_strength.setValue(int(float(hcfg.get('strength', 1.5)) * 100))
                self.s_erode.setValue(int(hcfg.get('erode_iter', 1)))
                self.s_dilate.setValue(int(hcfg.get('dilate_iter', 0)))
                self.s_feather.setValue(int(float(hcfg.get('feather_px', 2.0))))
                self.cb_guided.setChecked(bool(hcfg.get('use_guided', False)))
        except Exception:
            pass
        # 即時預覽事件綁定
        for s in (self.s_h, self.s_s, self.s_v, self.s_strength, self.s_erode, self.s_dilate, self.s_feather, self.wand_tol):
            s.valueChanged.connect(self._update_preview_if_live)
        if hasattr(self, 'cb_guided'):
            self.cb_guided.toggled.connect(self._update_preview_if_live)

    # ---------- 顯示/隱藏影片控制 ----------
    def _set_controls_visible(self, visible: bool):
        self.seek.setVisible(visible and _MULTIMEDIA_AVAILABLE)
        self.ctrl_bar.setVisible(visible and _MULTIMEDIA_AVAILABLE)
        self.rem_bg.setVisible(not visible)

    # ---------- file picking ----------
    def pick_file(self):
        dlg = QFileDialog(self, "選擇媒體")
        dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
        mimes = ["image/gif", "image/webp", "image/png", "image/jpeg"]
        if _MULTIMEDIA_AVAILABLE:
            mimes += [
                "video/mp4", "video/x-matroska", "video/quicktime",
                "video/x-msvideo", "video/webm"
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
                    e.acceptProposedAction(); return
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
                    self.toast.emit({
                        "level": "warn",
                        "title": "缺少多媒體相依",
                        "message": "影片播放需要系統安裝 PulseAudio / GStreamer（Linux/WSL）。已關閉影片功能。",
                        "duration": 5000,
                    })
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
                self._set_controls_visible(False)      # 靜態圖也隱藏
                # 保留原圖 pixmap 供預覽疊色使用
                self._orig_pix = QPixmap(pix)
                self.preview_img.setPixmap(self._scaled(self._orig_pix))
                # 記錄原始尺寸供座標轉換
                sz = reader.size(); self._last_img_size = (sz.width(), sz.height())
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
            orig_name = os.path.basename(self._current_path)
            base, ext = os.path.splitext(orig_name)
            text, ok = QInputDialog.getText(self, "儲存檔名", "輸入檔名（可不含副檔名）:", text=base)
            if not ok:
                return
            name = text.strip()
            if not name:
                return
            if os.path.splitext(name)[1] == "":
                name += ext
            out_dir = self._get_out_dir()
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, name)
            if os.path.abspath(save_path) == os.path.abspath(self._current_path):
                dlg = QMessageBox(self)
                dlg.setWindowTitle("提示")
                dlg.setText("來源與目標相同。")
                dlg.setIcon(QMessageBox.Icon.Information)
                # Apply local stylesheet so the dialog uses the same colors
                try:
                    dlg.setStyleSheet(_BASE)
                except Exception:
                    pass
                dlg.exec()
                return
            if os.path.exists(save_path):
                dlg = QMessageBox(self)
                dlg.setWindowTitle("覆寫確認")
                dlg.setText(f"檔案「{name}」已存在，是否覆寫？")
                dlg.setIcon(QMessageBox.Icon.Question)
                yes = dlg.addButton("是", QMessageBox.ButtonRole.YesRole)
                no = dlg.addButton("否", QMessageBox.ButtonRole.NoRole)
                # Apply local stylesheet so the dialog uses the same colors
                try:
                    dlg.setStyleSheet(_BASE)
                except Exception:
                    pass
                dlg.exec()
                if dlg.clickedButton() is not yes:
                    return
            if self._movie:
                self._movie.stop()

            p = name.split(".")[-1]

            if self.rem_bg.isChecked():

                os.makedirs('./temp', exist_ok=True)
                
                temp_path = os.path.join('./temp', name)

                
                shutil.copyfile(self._current_path, temp_path)

                if self._movie:
                    self._movie.start()
                # 依引擎決定 prefer 與參數
                prefer = getattr(self, "engine_box", None).currentText() if hasattr(self, "engine_box") else "hsv"
                opts = {}
                if prefer == "wand":
                    seed = getattr(self, "_wand_seed", None)
                    if seed is None:
                        self.toast.emit({"level":"warn","title":"魔術棒","message":"請先選取區域","duration":3000})
                        return
                    opts = {"seed": seed, "tolH": int(self.wand_tol.value()), "tolS": 60, "tolV": 60, "contiguous": True}
                elif prefer == "hsv":
                    opts = {"hsv": {"tol_h": int(self.s_h.value()), "tol_s": int(self.s_s.value()), "tol_v": int(self.s_v.value())}}
                self.removeBg.emit(temp_path, prefer, opts)
                self.toast.emit({
                    "level": "info",
                    "title": "remove backgrund...",
                    "message": f"Wait...",
                    "duration": 5000,
                })
                return
            if p in _EXTS:
                shutil.copyfile(self._current_path, save_path)
            else:
                #TODO convert to webp
                self.toast.emit({
                    "level": "warning",
                    "title": "不支援的格式",
                    "message": f"目前僅支援儲存為 GIF/WEBP。",
                    "duration": 5000,
                })
                pass
            if self._movie:
                self._movie.start()
            self.update_data.emit()
            self.toast.emit({
                "level": "info",
                "title": "儲存成功",
                "message": f"已儲存到 {os.path.relpath(save_path, os.getcwd())}",
                "duration": 5000,
            })
        except Exception as e:
            self.on_exception.emit(e)

    def _remove_bg(self):
        pass

    def eventFilter(self, obj, ev):
        # 魔術棒：在圖片模式、engine=wand 下，記錄點擊座標
        if obj is self.preview_img and ev.type() == QEvent.Type.MouseButtonPress:
            if hasattr(self, "engine_box") and self.engine_box.currentText() == "wand" and self._current_path and self.left_stack.currentIndex() == 0:
                pos = ev.position().toPoint()
                mapped = self._map_click_to_image(pos)
                if mapped is not None:
                    self._wand_seed = mapped  # (x,y)
                    self.toast.emit({
                        "level": "info", "title": "魔術棒", "message": f"取樣座標：{mapped}", "duration": 1500
                    })
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
        # 僅在圖片模式可用
        if not self._current_path:
            return
        from ui.wand_dialog import WandDialog
        dlg = WandDialog(self._current_path, tol_h=self.wand_tol.value(), parent=self)
        res = dlg.exec()
        out = dlg.get_result()
        if res and out:
            seed, tol_h = out
            self._wand_seed = seed
            self.wand_tol.setValue(int(tol_h))
            # 預覽
            self._update_preview()

    def _on_engine_changed(self, eng: str):
        is_hsv = (eng == "hsv")
        is_wand = (eng == "wand")
        for w in (self.s_h, self.s_s, self.s_v, self.s_strength, self.s_erode, self.s_dilate, self.s_feather, self.cb_guided):
            w.setEnabled(is_hsv)
        self.btn_wand.setEnabled(is_wand)
        self.wand_tol.setEnabled(is_wand)

    def _update_preview(self):
        try:
            if not self._current_path or self.left_stack.currentIndex() != 0:
                return
            import cv2, numpy as np
            from core.hsv_bg import compute_alpha
            from core.wand import compute_mask
            bgr = cv2.imread(self._current_path, cv2.IMREAD_COLOR)
            if bgr is None:
                return
            eng = self.engine_box.currentText() if hasattr(self, "engine_box") else "hsv"
            if eng == "hsv":
                alpha = compute_alpha(bgr, {"tol_h": int(self.s_h.value()), "tol_s": int(self.s_s.value()), "tol_v": int(self.s_v.value()), "strength": float(self.s_strength.value())/100.0, "erode_iter": int(self.s_erode.value()), "dilate_iter": int(self.s_dilate.value()), "feather_px": float(self.s_feather.value()), "use_guided": bool(self.cb_guided.isChecked())})
            elif eng == "wand":
                seed = getattr(self, "_wand_seed", None)
                if not seed:
                    self.toast.emit({"level":"warn","title":"魔術棒","message":"請先選取區域","duration":2000})
                    return
                alpha = compute_mask(bgr, seed, {"tolH": int(self.wand_tol.value()), "tolS": 60, "tolV": 60, "contiguous": True, "use_edge_barrier": True, "connectivity": 8})
            else:
                self.toast.emit({"level":"info","title":"預覽","message":"此引擎不支援即時預覽","duration":2000})
                return
            # 疊色到當前預覽大小
            ov = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8)
            ov[..., 1] = 255
            ov[..., 3] = (alpha > 0).astype(np.uint8) * 90
            from PyQt6.QtGui import QImage, QPainter
            qov = QImage(ov.data, ov.shape[1], ov.shape[0], 4 * ov.shape[1], QImage.Format.Format_RGBA8888)
            # 以原圖為底，避免覆疊多次後失真或透明
            base = getattr(self, "_orig_pix", None)
            if base is None or base.isNull():
                base = QPixmap(self._current_path)
                if base.isNull():
                    return
                self._orig_pix = QPixmap(base)
            scaled = self._scaled(base)
            qov = qov.scaled(scaled.size(), Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
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
            if hasattr(self, 'cb_live') and self.cb_live.isChecked():
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
        base = os.path.splitext(os.path.basename(self._current_path or "output"))[0]
        text, ok = QInputDialog.getText(self, "輸出檔名", "輸入檔名：", text=f"{base}{suffix}")
        if not ok or not text.strip():
            return None
        out_dir = os.path.join(os.getcwd(), "animes")
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, text.strip())

    def _ffmpeg(self, args: list[str]):
        try:
            proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True, proc.stderr.decode("utf-8", "ignore")
        except subprocess.CalledProcessError as e:
            self.on_exception.emit(Exception(e.stderr.decode("utf-8", "ignore")))
            return False, ""

    def _mark_in(self):
        if self.left_stack.currentIndex() == 1:
            self._in_ms = self.player.position()
            self.toast.emit({"level": "info", "title": "入點", "message": f"{self._in_ms/1000:.2f}s", "duration": 2000})

    def _mark_out(self):
        if self.left_stack.currentIndex() == 1:
            self._out_ms = self.player.position()
            self.toast.emit({"level": "info", "title": "出點", "message": f"{self._out_ms/1000:.2f}s", "duration": 2000})

    def _ffmpeg_trim(self):
        if not self._current_path or self._in_ms is None or self._out_ms is None or self._out_ms <= self._in_ms:
            return
        out = self._ask_output("_trim.mp4")
        if not out:
            return
        args = [
            "ffmpeg", "-y", "-ss", f"{self._in_ms/1000:.3f}", "-to", f"{self._out_ms/1000:.3f}",
            "-i", self._current_path, "-c", "copy", out,
        ]
        ok, _ = self._ffmpeg(args)
        if ok:
            self.toast.emit({"level": "info", "title": "完成", "message": f"已輸出 {os.path.basename(out)}", "duration": 4000})
            self.update_data.emit()

    def _ffmpeg_mute(self):
        if not self._current_path:
            return
        out = self._ask_output("_mute.mp4")
        if not out:
            return
        args = ["ffmpeg", "-y", "-i", self._current_path, "-c", "copy", "-an", out]
        ok, _ = self._ffmpeg(args)
        if ok:
            self.toast.emit({"level": "info", "title": "完成", "message": f"已輸出 {os.path.basename(out)}", "duration": 4000})
            self.update_data.emit()
