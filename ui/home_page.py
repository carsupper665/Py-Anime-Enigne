# ui/home_page.py
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QFileDialog, QSplitter,
    QInputDialog, QMessageBox, QSlider, QStackedLayout, QWidget as QW, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QUrl
from PyQt6.QtGui import QPixmap, QImageReader, QDragEnterEvent, QDropEvent, QMovie
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
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
_EXTS = ["gif", "webp", "png"]

class HomePage(QWidget):
    update_data = pyqtSignal()
    on_exception = pyqtSignal(object)
    fileSelected = pyqtSignal(str)
    toast = pyqtSignal(dict)  # {level,title,message,duration}
    removeBg = pyqtSignal(str, str)

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

        self.video_widget = QVideoWidget(self)
        self.player = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_out)
        self.player.setVideoOutput(self.video_widget)

        self.left_stack = QStackedLayout()
        self.left_stack.addWidget(self.preview_img)  # 0 = 圖片/GIF
        self.left_stack.addWidget(self.video_widget) # 1 = 影片

        # 進度條與控制列（影片）
        self.seek = QSlider(Qt.Orientation.Horizontal, self)
        self.seek.setEnabled(False)
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
        self.rem_bg.setStyleSheet("""
                                  font-weight: 300; font-size: 14px; color: #FFFFFF;
                                  """)
        rv.addWidget(self.rem_bg)
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

    # ---------- 顯示/隱藏影片控制 ----------
    def _set_controls_visible(self, visible: bool):
        self.seek.setVisible(visible)
        self.ctrl_bar.setVisible(visible)
        self.rem_bg.setVisible(not visible)

    # ---------- file picking ----------
    def pick_file(self):
        dlg = QFileDialog(self, "選擇媒體")
        dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
        dlg.setMimeTypeFilters([
            "image/gif", "image/webp", "image/png", "image/jpeg",
            "video/mp4", "video/x-matroska", "video/quicktime", "video/x-msvideo", "video/webm"
        ])
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
                # 影片模式
                self._clear_movie()
                self.player.setSource(QUrl.fromLocalFile(path))
                self.left_stack.setCurrentIndex(1)
                self.seek.setEnabled(False)
                self._set_controls_visible(True)
                self.player.play()
                return

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
            pix = QPixmap(path)
            if not pix.isNull():
                self.left_stack.setCurrentIndex(0)
                self._set_controls_visible(False)      # 靜態圖也隱藏
                self.preview_img.setPixmap(self._scaled(pix))
        except Exception as e:
            self.on_exception.emit(e)

    def clear_preview(self):
        self._current_path = None
        self._clear_movie()
        self.preview_img.clear()
        self.preview_img.setText("無預覽")
        self.player.stop()
        self.seek.setEnabled(False)
        self._set_controls_visible(False)
        self.save.setDisabled(True)

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
            out_dir = os.path.join(os.getcwd(), "animes")
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
                self.removeBg.emit(temp_path, p)
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
                "message": f"已儲存到 animes/{name}",
                "duration": 5000,
            })
        except Exception as e:
            self.on_exception.emit(e)

    def _remove_bg(self):
        pass

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