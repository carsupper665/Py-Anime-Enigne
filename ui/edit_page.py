# ui/edit_page.py

from __future__ import annotations
import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
    QFileDialog, QListWidget, QListWidgetItem,
)
from PyQt6.QtCore import (Qt, pyqtSignal, pyqtSlot, QSize,
                          )
from PyQt6.QtGui import (QPixmap, QMovie)
from logging import Logger
from .osd import OSD

BTN_STYLE = """
    QPushButton {
        background-color: %(btn_color)s;
        color: %(text_color)s;
        border: none;
        border-radius: 6px;
        padding: 6px 12px;
        width: 80px;
        font-size: 14px;
        font-family: "Source Han Sans TC"
        }
    QPushButton:hover {
        background-color: %(hover_color)s;
        }
    QPushButton:pressed {
        background-color: %(press_color)s;
        }
    QPushButton:disabled {
        background-color: #555;
        color: #888;
        }
"""

class EditPage(QWidget):
    on_exception = pyqtSignal(Exception)
    toast = pyqtSignal(dict) # payload: {level, title, message, duration}
    sync_gifs = pyqtSignal(dict)
    def __init__(self, parent=None, logger: Logger=None, activated_gifs: dict[str:QWidget] | None = None):
        super().__init__(parent)
        self.setObjectName("EditPage")
        self.logger = logger
        self.activated_gifs = activated_gifs or {}
        self._movie = None

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Layout Editor", self)
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        title.setStyleSheet("""
            background-color: none;
            font-family: "Source Han Sans TC";
            font-weight: 600;
            font-size: 30px;
            color: #FFFFFF;
        """)

        v.addWidget(title)
        row = QHBoxLayout()
        self.g_list = QListWidget(self)
        self.g_list.setStyleSheet("""
            QListWidget { background:#111; border:1px solid #333; border-radius:8px; color:#ddd; }
            QListWidget::item { padding:6px 10px; }
            QListWidget::item:selected { background:#2a2a2a; }
        """)
        self.g_list.itemDoubleClicked.connect(self._show_selected)
        rv = QVBoxLayout()
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(6)
        btns = QHBoxLayout()
        btns.setContentsMargins(0, 0, 0, 0)
        btns.setSpacing(6)

        self.preview = QLabel(self)
        self.preview.setMinimumSize(480, 270)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("""
            QLabel {
                background-color: #111;
                border: 1px solid #333;
                border-radius: 12px;
            }
        """)
        self.hide_btn = QPushButton("hide", self)
        self.hide_btn.setStyleSheet(BTN_STYLE % {"btn_color": "#2c7ce5", "text_color": "#FFF",
                                                "hover_color": "#3c8ce5", "press_color": "#1c6ce5"})
        self.close_btn = QPushButton("close", self)
        self.close_btn.setStyleSheet(BTN_STYLE % {"btn_color": "#e52c2c", "text_color": "#EEE",
                                                 "hover_color": "#e53c3c", "press_color": "#e51c1c"})
        self.hide_btn.setEnabled(False)
        self.close_btn.setEnabled(False)
        self.hide_btn.clicked.connect(self._hide_selected)
        self.close_btn.clicked.connect(self._close_selected)
        btns.addWidget(self.hide_btn, alignment=Qt.AlignmentFlag.AlignRight)
        btns.addWidget(self.close_btn, alignment=Qt.AlignmentFlag.AlignRight)
        rv.addWidget(self.preview)
        rv.addStretch(1)

        rv.addLayout(btns)
        row.addWidget(self.g_list)
        row.addLayout(rv)
        # row.addWidget(self.preview)

        v.addLayout(row)
    
    def set_btns_enabled(self,):
        self.hide_btn.setEnabled(not self.hide_btn.isEnabled())
        self.close_btn.setEnabled(not self.close_btn.isEnabled())

    def _sync(self):
        self.sync_gifs.emit(self.activated_gifs)
        self.logger.debug(f"syncing...")

    def update_data(self, payload: dict | str, mode: str = "replace"):
        """
        更新內部資料，並刷新顯示。
        mode: replace | append
        """
        if isinstance(payload, str) and mode == "append":
            self.g_list.addItem(QListWidgetItem(payload))
            return
        
        if isinstance(payload, str) and mode == "del":
            items = self.g_list.findItems(payload, Qt.MatchFlag.MatchExactly)
            for it in items:
                self.g_list.takeItem(self.g_list.row(it))
            return
            
        names = [k for k in payload.keys()]
        if mode == "replace":
            self.g_list.clear()

        for name in names:
            it = QListWidgetItem(name)
            self.g_list.addItem(it)

    def _fit_size(self, box: QSize) -> QSize:
        return QSize(max(1, box.width()-16), max(1, box.height()-16))
    
    def _set_movie(self, movie: QMovie):
        self._clear_movie()
        self._movie = movie
        self.preview.setMovie(self._movie)
        self._movie.setScaledSize(self._fit_size(self.preview.size()))
        self._movie.start()

    def _clear_movie(self):
        if self._movie:
            self._movie.stop()
            self._movie.deleteLater()
            self._movie = None
            self.preview.clear()

    def _scaled(self, pix: QPixmap) -> QPixmap:
        return pix.scaled(
            self.preview.size() - QSize(16, 16),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
    
    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._movie:
            self._movie.setScaledSize(self._fit_size(self.preview.size()))
        elif self.preview.pixmap() is not None:
            self.preview.setPixmap(self._scaled(self.preview.pixmap()))

    def _close_selected(self):
        self.set_btns_enabled()
        item = self.g_list.currentItem()
        self._clear_movie()
        osd: OSD = self.activated_gifs[item.text()]
        if osd is not None:
            osd.osd_close()
            self.logger.debug(f"Close selected gif: {item.text()}")
        else:
            self.logger.warning(f"Tried to close non-existing gif: {item.text()}")
            self.toast.emit({
                "level": 'warn',
                "title": "關閉失敗",
                "message": f"找不到名稱為 {item.text()} 的 GIF。",
                "duration": 3000
            })

    def _hide_selected(self):
        self.set_btns_enabled()
        item = self.g_list.currentItem()
        self._clear_movie()
        osd: OSD = self.activated_gifs[item.text()]
        if osd is not None:
            osd.osd_hide()
            self.logger.debug(f"Hide selected gif: {item.text()}")
        else:
            self.logger.warning(f"Tried to hide non-existing gif: {item.text()}")
            self.toast.emit({
                "level": 'warn',
                "title": "隱藏失敗",
                "message": f"找不到名稱為 {item.text()} 的 GIF。",
                "duration": 3000
            })

    @pyqtSlot()
    def _show_selected(self):
        osd: OSD = None
        item = self.g_list.currentItem()
        self._clear_movie()
        osd = self.activated_gifs[item.text()]
        osd.show()
        osd.activateWindow()
        osd.enter_edit_mode()
        path = osd.get_path()
        self.logger.debug(f"Show selected gif_path: {path}")
        m = QMovie(path)
        self._set_movie(m)
        self.set_btns_enabled()

    @pyqtSlot(QWidget)
    def _update_list(self, w: QWidget):
        self.activated_gifs[w.objectName()] = w
        self.update_data(w.objectName(), mode="append")
        self.logger.debug(f"activated_gifs updated: {self.activated_gifs}")
        self._sync()

    @pyqtSlot(str)
    def _delete_gif(self, name: str):
        try:
            self.activated_gifs.pop(name)
            self.update_data(name, mode="del")
            self._sync()
        except KeyError:
            self.logger.warning(f"Tried to delete non-existing gif {name}.")
            self.toast.emit({
                "level": 'warn',
                "title": "刪除失敗",
                "message": f"找不到名稱為 {name} 的 GIF。",
                "duration": 3000
            })
        except Exception as e:
            self.on_exception.emit(e)
