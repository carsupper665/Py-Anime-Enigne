#main.py
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, 
    QStackedWidget, QApplication, QLabel
)
from PyQt6.QtGui import (QFont,)
from PyQt6.QtCore import (Qt, pyqtSlot, QThread,)
import sys
from ui import *

import sys

class Main(QMainWindow):
    def __init__(self, level: str | int="DEBUG", save_log: bool=False, scale: int=70):
        super().__init__()
        self.logger = loggerFactory(logger_name="PyAnimeEngine", log_level=level, write_log=save_log).getLogger()
        self.logger.info("⏳Starting up...")
        self.setObjectName(self.__class__.__name__)
        self.setWindowTitle("Anime Engine")
        self.resize(16*scale, 9*scale)
        self.setStyleSheet("background-color: #212121;")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self.anime_data = None

        self.toast = Toast(self)
        self.toast_loading = LoadingToast(self)
        self.toast.fatalTriggered.connect(self._clear_all)

        try:

            self.main_widget = QWidget()

            self.main_layout = QHBoxLayout(self.main_widget)
            self.main_layout.setContentsMargins(0, 0, 0, 0)

            self.nav_bar = NavBar(self)
            self.main_layout.addWidget(self.nav_bar)

            self.content_widget = QStackedWidget()
            self.main_layout.addWidget(self.content_widget)

            self.load_pages()

            self.set_workers()

            self.nav_bar.tabChanged.connect(self.content_widget.setCurrentIndex)
            self.nav_bar.setDisabled(True)

            self.setCentralWidget(self.main_widget)
            
            self.logger.info("Setup wrapped. Game on.")

        except Exception as e:
            self._on_exception(e)

    def set_workers(self):
        self.gif_loader = GifLoader(self.logger)
        self.gif_loader.reload_finished.connect(self.reload_anime_data)
        self.gif_loader.error.connect(self._on_exception)
        self.home_page.update_data.connect(self.gif_loader.reload)
        self.home_page.removeBg.connect(self._on_rmbg)

        self.rmbg_thread = RmbgThread(self.logger)
        self.rmbg_thread.progress.connect(self.toast_loading.update_progress)
        self.rmbg_thread.error.connect(self._on_exception)
        self.rmbg_thread.finished.connect(self._rmbg_finished)

    @pyqtSlot(str, str) # (src_path: str, prefer: str = "auto"):
    def _on_rmbg(self, src: str, prefer: str):
        self.toast_loading.show_loading(title="Removeing BG")
        self.logger.debug("Removeing BG")
        if prefer == "":
            prefer = "auto"
        self.rmbg_thread.remove_bg(src_path=src, prefer=prefer)

    @pyqtSlot(dict) # {input, output, kind, frames?, manifest?}
    def _rmbg_finished(self, payload: dict):
        _input = payload.get("input"); _output = payload.get("output"); _kind = payload.get("kind"); _frames = payload.get("frames", None); _manifest = payload.get("manifest", None)
        self.logger.debug(f"input: {_input}\noutput:{_output}\nkind{_kind}")
        self.logger.debug(f"frames: {_frames}, manifest{_manifest}")
        self.toast.show_notice(INFO, title="File Saved", message=f"input: {_input}\noutput:{_output}\nkind{_kind}", px=self._get_x(), py=self._get_y())
        self.gif_loader.reload()

    @pyqtSlot(dict)
    def reload_anime_data(self, data):
        self.anime_data = data
        self.logger.debug(f"Anime data reloaded")
        self.logger.debug(f"Data: {self.anime_data}")
        self.add_page.update_data(data, mode="replace")

    def load_pages(self):
        self.loading_page = LoadingPage(logger=self.logger)
        self.home_page = HomePage(self)
        self.add_page = AddPage(self, self.logger)
        self.edit_page = EditPage(self, self.logger)
        self.add_page.on_activated.connect(self.edit_page._update_list)
        self.add_page.gif_closed.connect(self.edit_page._delete_gif)
        self.edit_page.sync_gifs.connect(self.add_page.on_sync)
        pages = [self.home_page, self.add_page, self.edit_page,self.loading_page]

        for page in pages:
            self.content_widget.addWidget(page)
            if hasattr(page, "on_exception"):
                page.on_exception.connect(self._on_exception)
            if hasattr(page, "toast"):
                page.toast.connect(self._toast)
        self.content_widget.setCurrentIndex(len(pages)-1)

        self.loading_thread = QThread()
        self.load = LoadThread(self.logger)
        self.load.moveToThread(self.loading_thread)
        self.loading_thread.started.connect(self.load.load)
        self.loading_thread.finished.connect(self.loading_thread.quit)
        self.load.progress.connect(self.loading_page.update_progress)
        self.load.finished.connect(self._init_finished)
        self.loading_thread.start()


    @pyqtSlot(dict)
    def _toast(self, payload: dict):
        level = payload.get("level", INFO)
        title = payload.get("title", "Notification")
        message = payload.get("message", "")
        duration = payload.get("duration", 3000)
        self.toast.show_notice(level, title, message, duration, px=self._get_x(), py=self._get_y())

    def _clear_all(self):
        self.logger.info("stopping all threads...")
        pass

    def _get_x(self):
        return self.pos().x()
    
    def _get_y(self):
        return self.pos().y()
    
    @pyqtSlot(dict)
    def _init_finished(self, payload: dict):
        self.loading_thread.quit()
        self.loading_thread.wait()
        self.toast.show_notice(INFO, "Welcome!", 
            message="Welcome to the PyAnime Engine.", 
            )
        self.nav_bar.setEnabled(True)
        self.content_widget.setCurrentIndex(0)
        self.anime_data = payload
        self.add_page.update_data(payload, mode="replace")
        self.logger.debug(f"Initial data loaded: {self.anime_data}")
        # self.toast.show_loading("Initializing...")

    @pyqtSlot(Exception)
    def _on_exception(self,e: Exception):
        self.toast_loading.on_exception_cancel()
        if isinstance(e, FFmpegNotFoundError):
            self.toast.show_notice(ERROR, "FFmpeg Not Found", str(e), 10000, px=self._get_x(), py=self._get_y())
            return
        self.hide()
        import traceback
        exctype = type(e)
        tb_text = "".join(traceback.TracebackException.from_exception(e).format())
        title ="App Crash Exception"
        self.toast.show_notice(FATAL, title, e, 60000, traceback=tb_text)
        sys.__excepthook__(exctype, e, traceback)

def main():
    # app = QApplication(sys.argv)
    app = MainApp(sys.argv)
    install_global_handlers(app)
    connect_crash_dialog(app)
    app.setFont(QFont(r'./src/fonts'))
    window = Main()
    window.show()

    sys.exit(app.exec())

if __name__ == '__main__':
    main()

