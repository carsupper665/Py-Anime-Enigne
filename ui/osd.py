# ui/osd.py
from PyQt6.QtCore import Qt, QPoint, QSize, pyqtSignal
from PyQt6.QtWidgets import QWidget, QLabel, QMenu, QSizeGrip
from PyQt6.QtGui import QMovie, QPainter, QPen, QColor, QImageReader

class OSD(QWidget):
    on_exception = pyqtSignal(Exception)
    on_closed = pyqtSignal(str)  # name

    def __init__(self, parent=None, name: str = "", gif_path: str | None = None, size: tuple[int, int] | None = None):
        super().__init__(parent, flags=Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._movie: QMovie | None = None
        self._drag_pos: QPoint | None = None
        self._grips: list[QSizeGrip] = []
        self.gif_path: str | None = None
        self._edit_mode = False
        self._content_size = QSize(320, 240)  # 來源等比基準

        self.setObjectName(name)
        self.setWindowTitle(name)
        self.setMinimumSize(200, 200)  # 最小 200px

        self.label = QLabel(self)
        self.label.setStyleSheet("background: transparent;")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.resize(320, 240)

        if gif_path:
            self.set_media(gif_path, size)
            
    def get_path(self) -> str | None:
        return self.gif_path

    # ===== 編輯模式 =====
    def enter_edit_mode(self):
        self._edit_mode = True
        self._ensure_grips()
        for g in self._grips: g.show()
        self.update(); self.raise_(); self.activateWindow()

    def exit_edit_mode(self):
        self._edit_mode = False
        for g in self._grips: g.hide()
        self.update()

    def _ensure_grips(self):
        if self._grips: return
        self._grips = [QSizeGrip(self) for _ in range(4)]
        for g in self._grips: g.setFixedSize(12, 12)
        self._relayout_grips()

    def _relayout_grips(self):
        if not self._grips: return
        m, s = 1, 12
        w, h = self.width(), self.height()
        self._grips[0].move(m, m)               # 左上
        self._grips[1].move(w - m - s, m)       # 右上
        self._grips[2].move(m, h - m - s)       # 左下
        self._grips[3].move(w - m - s, h - m - s)  # 右下

    # ===== 媒體與縮放 =====
    def set_media(self, path: str, size: tuple[int, int] | None = None, max_size: tuple[int, int] | None = (800, 800)):
        self.gif_path = path
        try:
            if self._movie:
                self._movie.stop()
                self._movie.deleteLater()

            # 讀原始尺寸作為等比基準
            if size:
                w0, h0 = size
            else:
                r = QImageReader(path)              # 取來源邏輯尺寸
                s = r.size()
                w0, h0 = (s.width() or 320), (s.height() or 240)

            # 先做上限裁切
            if max_size:
                mw, mh = max_size
                k = min(mw / w0, mh / h0, 1.0)
                w0, h0 = int(w0 * k), int(h0 * k)

            self._content_size = QSize(w0, h0)

            m = QMovie(path)
            self.label.setMovie(m)
            self._movie = m
            self._apply_scaled_size()              # 依當前視窗做等比縮放
            m.start()
        except Exception as e:
            self.on_exception.emit(e)

    def _apply_scaled_size(self):
        """依視窗大小，把動畫等比縮放到可用區域，並保底最小 200。"""
        if not self._movie:
            return
        box = self.size()
        cw, ch = self._content_size.width(), self._content_size.height()
        if cw <= 0 or ch <= 0:
            cw, ch = 320, 240

        # 等比填滿於視窗（保留邊距可自行調整）
        k = min(box.width() / cw, box.height() / ch)
        w, h = max(200, int(cw * k)), max(200, int(ch * k))
        self._movie.setScaledSize(QSize(w, h))     # 核心：隨視窗更新 scaledSize。:contentReference[oaicite:1]{index=1}
        self.label.resize(self.size())

    # ===== 右鍵、拖曳、事件 =====
    def contextMenuEvent(self, e):
        menu = QMenu(self)
        act_edit = menu.addAction("切換編輯模式")
        act_hide = menu.addAction("隱藏")
        act_close = menu.addAction("關閉")
        chosen = menu.exec(e.globalPos())
        if chosen == act_edit:
            (self.exit_edit_mode() if self._edit_mode else self.enter_edit_mode())
        elif chosen == act_hide:
            self.hide()
        elif chosen == act_close:
            self.close()
            self.on_closed.emit(self.objectName())

    def osd_close(self):
        self.close()
        self.on_closed.emit(self.objectName())
    
    def osd_hide(self):
        self.hide()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and not self._edit_mode:
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag_pos is not None and e.buttons() & Qt.MouseButton.LeftButton and not self._edit_mode:
            self.move(e.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, e):
        self._drag_pos = None

    def focusOutEvent(self, e):
        if self._edit_mode:
            self.exit_edit_mode()
        super().focusOutEvent(e)

    def resizeEvent(self, ev):
        self._relayout_grips()
        self._apply_scaled_size()                  # 視窗改變時同步縮放
        super().resizeEvent(ev)

    def paintEvent(self, ev):
        if self._edit_mode:
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setPen(QPen(QColor("#00ff66"), 2))
            p.drawRect(self.rect().adjusted(1, 1, -1, -1))
        super().paintEvent(ev)
