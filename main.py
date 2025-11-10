# main.py
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QStackedWidget,
    QApplication,
    QLabel,
)
from PyQt6.QtGui import (
    QFont,
)
from PyQt6.QtCore import (
    Qt,
    pyqtSlot,
    QThread,
)
import logging
import sys
from ui import *
from core.config import load_config, save_config
from core.export_context import (
    ExportContextError,
    build_export_context,
)
from core.diagnostics import (
    attach_diagnostic,
    format_user_message,
    generate_diagnostic_id,
    log_structured,
)
import os


class Main(QMainWindow):
    def __init__(
        self, level: str | int = "DEBUG", save_log: bool = False, scale: int = 70
    ):
        super().__init__()
        self.logger = loggerFactory(
            logger_name="PyAnimeEngine", log_level=level, write_log=save_log
        ).getLogger()
        self.logger.info("⏳Starting up...")
        self.setObjectName(self.__class__.__name__)
        self.setWindowTitle("Anime Engine")
        self.resize(16 * scale, 9 * scale)
        self.setStyleSheet("background-color: #212121;")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self.anime_data = None
        # 讀取設定
        self.config = load_config()

        self.toast = Toast(self)
        self.toast_loading = LoadingToast(self)
        self.toast.fatalTriggered.connect(self._clear_all)
        self._suppress_rmbg_loading = False

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
        # 舊路徑相容：HomePage.removeBg 也改走佇列（非阻塞統一，OpenSpec update-queue-nonblocking-integration）
        self.home_page.removeBg.connect(self._on_enqueue_job_via_removebg)

        self.rmbg_thread = RmbgThread(self.logger)
        self.rmbg_thread.progress.connect(self.toast_loading.update_progress)
        self.rmbg_thread.error.connect(self._on_exception)
        self.rmbg_thread.finished.connect(self._rmbg_finished)

        # Queue: 處理佇列
        self.queue = ProcessingQueue()
        # 由 HomePage 送進來的加入佇列
        if hasattr(self.home_page, "enqueueJob"):
            self.home_page.enqueueJob.connect(self._on_enqueue_job)
        # 佇列開始某個工作 → 顯示 Loading 並呼叫既有去背流程
        self._queue_started_count = 0
        self.queue.job_started.connect(self._on_queue_job_started)
        # 將 worker 訊號轉發給佇列（以便 UI 顯示）
        self.rmbg_thread.progress.connect(self.queue.notify_progress)
        self.rmbg_thread.finished.connect(self.queue.notify_finished)
        self.rmbg_thread.error.connect(self.queue.notify_error)
        # 佇列計數與狀態顯示
        self._queue_pending = 0
        self.queue.job_enqueued.connect(lambda _j: self._update_queue_ui(delta=1))
        self.queue.job_finished.connect(lambda _j, _p: self._update_queue_ui(delta=-1))
        self.queue.job_error.connect(lambda _j, _e: self._update_queue_ui(delta=-1))
        self.queue.queue_empty.connect(lambda: self._set_queue_label(0))
        self.queue.queue_empty.connect(self._on_queue_empty)
        # 任務完成/錯誤時保險關閉 Loading（通常 100% 會自動關閉）
        self.queue.job_finished.connect(
            lambda *_: self.toast_loading.on_exception_cancel()
        )
        self.queue.job_error.connect(
            lambda *_: self.toast_loading.on_exception_cancel()
        )

    @pyqtSlot(str, str, dict)  # (src_path: str, prefer: str = "auto", opts: dict):
    def _on_rmbg(self, src: str, prefer: str, opts: dict):
        if getattr(self, "_suppress_rmbg_loading", False):
            # 僅抑制一次，避免重覆顯示
            self._suppress_rmbg_loading = False
        else:
            self.toast_loading.show_loading(title="Removeing BG")
        payload_opts = dict(opts or {})
        diagnostic_id = payload_opts.get("diagnostic_id") or generate_diagnostic_id()
        payload_opts["diagnostic_id"] = diagnostic_id
        log_structured(
            self.logger,
            logging.INFO,
            diagnostic_id,
            "rmbg.request",
            src=src,
            prefer=prefer,
        )
        try:
            context = build_export_context(src, prefer, payload_opts, self.config)
        except ExportContextError as err:
            self._on_exception(err)
            return
        # 設定環境變數（目前僅 OpenVINO 模型路徑）
        for key, value in context.env.items():
            if value:
                os.environ[key] = value
        request = context.request

        self.rmbg_thread.remove_bg(
            src_path=request.src_path,
            out_dir=request.output_dir,
            prefer=request.prefer,
            engine=request.engine,
            hsv_cfg=request.hsv_config,
            wand=request.wand_config,
            image_format=request.image_format,
            anim_format=request.anim_format,
            range_ms=request.range_ms,
            export_opts=request.options.to_dict(),
            diagnostic_id=diagnostic_id,
        )

    # ----- Queue + LoadingToast integration -----
    @pyqtSlot(object)
    def _on_queue_job_started(self, job):
        try:
            # i/N：以「目前待處理 + 1」作為 N；i 以批次計數累加
            self._queue_started_count = getattr(self, "_queue_started_count", 0) + 1
            total = max(1, getattr(self, "_queue_pending", 0) + 1)
            title = f"Processing {self._queue_started_count}/{total}"
            msg = os.path.basename(getattr(job, "src", ""))
            diagnostic_id = getattr(job, "diagnostic_id", None)
            if not diagnostic_id and isinstance(getattr(job, "opts", None), dict):
                diagnostic_id = job.opts.get("diagnostic_id")
            if diagnostic_id:
                msg = f"{msg} (ID {diagnostic_id})"
            log_structured(
                self.logger,
                logging.INFO,
                diagnostic_id,
                "queue.job.started",
                src=getattr(job, "src", ""),
                prefer=getattr(job, "prefer", ""),
            )
            # 抑制 _on_rmbg 再次顯示 loading
            self._suppress_rmbg_loading = True
            self.toast_loading.show_loading(title=title, message=msg)
            # 開始執行
            self._on_rmbg(job.src, job.prefer, job.opts)
        except Exception as e:
            self._on_exception(e)

    @pyqtSlot()
    def _on_queue_empty(self):
        try:
            self._queue_started_count = 0
            # 確保 Loading 關閉
            self.toast_loading.on_exception_cancel()
        except Exception:
            pass

    @pyqtSlot(dict)  # {input, output, kind, frames?, manifest?}
    def _rmbg_finished(self, payload: dict):
        _input = payload.get("input")
        _output = payload.get("output")
        _kind = payload.get("kind")
        _frames = payload.get("frames", None)
        _manifest = payload.get("manifest", None)
        diagnostic_id = payload.get("diagnostic_id")
        log_structured(
            self.logger,
            logging.INFO,
            diagnostic_id,
            "rmbg.finished",
            input=_input,
            output=_output,
            kind=_kind,
            frames=_frames,
        )
        self.gif_loader.reload()
        message = f"input: {_input}\noutput:{_output}\nkind{_kind}"
        if diagnostic_id:
            message = f"{message}\n追蹤：{diagnostic_id}"
        self.toast.show_notice(
            INFO,
            title="File Saved",
            message=message,
            px=self._get_x(),
            py=self._get_y(),
        )

    @pyqtSlot(dict)
    def reload_anime_data(self, data):
        self.anime_data = data
        self.logger.debug(f"Anime data reloaded")
        self.logger.debug(
            f"Data: {self.anime_data if len(self.anime_data.get('items', [])) < 5 else '...'}"
        )
        self.add_page.update_data(data, mode="replace")

    def load_pages(self):
        self.loading_page = LoadingPage(logger=self.logger)
        self.home_page = HomePage(self)
        self.add_page = AddPage(self, self.logger)
        self.edit_page = EditPage(self, self.logger)
        self.settings_page = SettingsPage(self)
        # 設定儲存後更新主設定
        self.settings_page.configSaved.connect(self._reload_config)
        self.add_page.on_activated.connect(self.edit_page._update_list)
        self.add_page.gif_closed.connect(self.edit_page._delete_gif)
        self.edit_page.sync_gifs.connect(self.add_page.on_sync)
        pages = [
            self.home_page,
            self.add_page,
            self.edit_page,
            self.settings_page,
            self.loading_page,
        ]

        for page in pages:
            self.content_widget.addWidget(page)
            if hasattr(page, "on_exception"):
                page.on_exception.connect(self._on_exception)
            if hasattr(page, "toast"):
                page.toast.connect(self._toast)
        self.content_widget.setCurrentIndex(len(pages) - 1)

        self.loading_thread = QThread()
        self.load = LoadThread(self.logger)
        self.load.moveToThread(self.loading_thread)
        self.loading_thread.started.connect(self.load.load)
        self.loading_thread.finished.connect(self.loading_thread.quit)
        self.load.progress.connect(self.loading_page.update_progress)
        self.load.finished.connect(self._init_finished)
        self.loading_thread.start()

    def _reload_config(self, cfg: dict):
        self.logger.info("設定已更新，重新載入設定…")
        self.config = cfg

    @pyqtSlot(dict)
    def _toast(self, payload: dict):
        level = payload.get("level", INFO)
        title = payload.get("title", "Notification")
        message = payload.get("message", "")
        duration = payload.get("duration", 3000)
        self.toast.show_notice(
            level, title, message, duration, px=self._get_x(), py=self._get_y()
        )

    def _clear_all(self):
        self.logger.info("stopping all threads...")
        pass

    def _get_x(self):
        return self.pos().x()

    def _get_y(self):
        return self.pos().y()

    # ---- OSD layout persistence ----
    def _save_osd_layout(self):
        try:
            osd_list = []
            # Collect from EditPage's activated_gifs
            if hasattr(self, "edit_page") and hasattr(self.edit_page, "activated_gifs"):
                for name, osd in getattr(self.edit_page, "activated_gifs", {}).items():
                    try:
                        if osd is None:
                            continue
                        g = osd.geometry()
                        path = osd.get_path() if hasattr(osd, "get_path") else None
                        if not path:
                            continue
                        osd_list.append(
                            {
                                "name": name,
                                "path": path,
                                "x": int(g.x()),
                                "y": int(g.y()),
                                "w": int(g.width()),
                                "h": int(g.height()),
                                "visible": bool(osd.isVisible()),
                            }
                        )
                    except Exception:
                        continue
            self.config["osd"] = osd_list
            save_config(self.config)
        except Exception as e:
            self.logger.error(f"save osd layout failed: {e}")

    def _restore_osd_layout(self):
        try:
            items = self.config.get("osd", []) if isinstance(self.config, dict) else []
            if not items:
                return
            for it in items:
                try:
                    name = it.get("name") or ""
                    path = it.get("path") or ""
                    if not path:
                        continue
                    osd = OSD(name=name)
                    # Wire events to existing pages
                    if hasattr(self.add_page, "gif_closed"):
                        osd.on_closed.connect(self.add_page.gif_closed)
                    if hasattr(self.add_page, "on_exception"):
                        osd.on_exception.connect(self.add_page.on_exception)
                    osd.set_media(path)
                    # Geometry
                    x = int(it.get("x", 100))
                    y = int(it.get("y", 100))
                    w = int(it.get("w", 320))
                    h = int(it.get("h", 240))
                    osd.resize(max(200, w), max(200, h))
                    osd.move(max(0, x), max(0, y))
                    if it.get("visible", True):
                        osd.show()
                        osd.activateWindow()
                    # Sync into lists via AddPage
                    if hasattr(self.add_page, "set_activated_gifs"):
                        self.add_page.set_activated_gifs(osd)
                except Exception as ie:
                    self.logger.error(f"restore osd failed: {ie}")
        except Exception as e:
            self.logger.error(f"restore osd layout failed: {e}")

    @pyqtSlot(dict)
    def _close_all_osd(self):
        try:
            if hasattr(self, "edit_page") and hasattr(self.edit_page, "activated_gifs"):
                try:
                    items = list(self.edit_page.activated_gifs.items())
                except Exception:
                    items = []
                for name, osd in items:
                    try:
                        if osd is None:
                            continue
                        osd.close()
                    except Exception:
                        pass
        except Exception:
            pass

    def closeEvent(self, ev):
        try:
            self._save_osd_layout()
        except Exception:
            pass
        try:
            self._close_all_osd()
        except Exception:
            pass
        super().closeEvent(ev)

    def _init_finished(self, payload: dict):
        self.loading_thread.quit()
        self.loading_thread.wait()
        self.toast.show_notice(
            INFO,
            "Welcome!",
            message="Welcome to the PyAnime Engine.",
        )
        self.nav_bar.setEnabled(True)
        self.content_widget.setCurrentIndex(0)
        self.anime_data = payload
        self.add_page.update_data(payload, mode="replace")
        self.logger.debug(f"Initial data loaded: {self.anime_data}")
        # self.toast.show_loading("Initializing...")
        try:
            self._restore_osd_layout()
        except Exception:
            pass

    # ----- ProcessingQueue integration -----
    @pyqtSlot(str, str, dict)
    def _on_enqueue_job(self, src: str, prefer: str, opts: dict):
        try:
            payload_opts = dict(opts or {})
            # 可選：持久化最近的 HSV 設定
            if prefer == "hsv" and "hsv" in payload_opts:
                hcfg = self.config.get("hsv", {})
                hcfg.update(payload_opts["hsv"])
                self.config["hsv"] = hcfg
                # OpenSpec: add-processing-queue — persist last engine options
                # spec: openspec/changes/add-processing-queue/tasks.md:1
                try:
                    save_config(self.config)
                except Exception:
                    pass
            diagnostic_id = (
                payload_opts.get("diagnostic_id") or generate_diagnostic_id()
            )
            payload_opts["diagnostic_id"] = diagnostic_id
            # 入列
            self.queue.enqueue(
                QueueJob(
                    src=src,
                    prefer=prefer or "auto",
                    opts=payload_opts,
                    diagnostic_id=diagnostic_id,
                )
            )
            message = os.path.basename(src)
            if diagnostic_id:
                message = f"{message}\n追蹤：{diagnostic_id}"
            log_structured(
                self.logger,
                logging.INFO,
                diagnostic_id,
                "queue.enqueue",
                src=src,
                prefer=prefer or "auto",
            )
            self.toast.show_notice(
                INFO, "已加入佇列", message, px=self._get_x(), py=self._get_y()
            )
        except Exception as e:
            self._on_exception(e)

    @pyqtSlot(str, str, dict)
    def _on_enqueue_job_via_removebg(self, src: str, prefer: str, opts: dict):
        # OpenSpec: update-queue-nonblocking-integration — redirect direct removeBg to queue
        self._on_enqueue_job(src, prefer, opts)

    def _set_queue_label(self, n: int):
        self._queue_pending = max(0, n)
        try:
            if hasattr(self.home_page, "queue_label"):
                self.home_page.queue_label.setText(
                    f"Queue: {self._queue_pending} pending"
                )
        except Exception:
            pass

    def _update_queue_ui(self, delta: int = 0):
        self._set_queue_label(self._queue_pending + delta)

    @pyqtSlot(Exception)
    def _on_exception(self, e: Exception):
        self.toast_loading.on_exception_cancel()
        diagnostic_id = getattr(e, "diagnostic_id", None)
        user_message = format_user_message(str(e), diagnostic_id)
        log_structured(
            self.logger,
            logging.ERROR,
            diagnostic_id,
            "app.exception",
            error=user_message,
        )
        if isinstance(e, FFmpegNotFoundError):
            self.toast.show_notice(
                ERROR,
                "FFmpeg Not Found",
                user_message,
                10000,
                px=self._get_x(),
                py=self._get_y(),
            )
            return
        self.hide()
        import traceback

        exctype = type(e)
        tb_text = "".join(traceback.TracebackException.from_exception(e).format())
        title = "App Crash Exception"
        self.toast.show_notice(FATAL, title, user_message, 60000, traceback=tb_text)
        sys.__excepthook__(exctype, e, traceback)


def main():
    # app = QApplication(sys.argv)
    app = MainApp(sys.argv)
    install_global_handlers(app)
    connect_crash_dialog(app)
    app.setFont(QFont(r"./src/fonts"))
    window = Main(level="DEBUG")
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
