#ui/ui_error.py
class UnexpectedError(Exception):
    def __init__(self, message="An unexpected error occurred. Please try reinstalling the application or contact support if the issue persists."):
        self.message = message
        super().__init__(self.message)

import sys, os, threading, asyncio, traceback, datetime, faulthandler
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QObject, pyqtSignal, Qt, qInstallMessageHandler
from PyQt6.QtGui import QIcon
from .toast import Toast

LOG_DIR = os.path.join(os.getcwd(), "crash_logs")
os.makedirs(LOG_DIR, exist_ok=True)

def _now():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def _log_path():
    return os.path.join(LOG_DIR, f"crash_{_now()}.log")

def _write_log(title: str, tb: str) -> str:
    path = _log_path()
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"[{title}]\n")
        f.write(tb)
    return path

class ErrorBus(QObject):
    error = pyqtSignal(str, str, Exception)  # title, traceback, exception instance

class CrashApp(QApplication):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.error_bus = ErrorBus()
        self._guard = False  # 防止重入

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except BaseException as e:
            self._emit("Qt event error", e)
            return False

    def _emit(self, title: str, e: BaseException):
        if self._guard:
            return
        self._guard = True
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        self.error_bus.error.emit(title, tb, e)
        self._guard = False

def install_global_handlers(app: CrashApp):
    # 1) Python 主執行緒
    def sys_hook(exctype, exc, tb):
        app._emit("Uncaught exception", exc)
    sys.excepthook = sys_hook

    # 2) 未可拋出的例外（Python 3.8+）
    def unraisable_hook(unraisable):
        tb = "".join(traceback.format_exception_only(type(unraisable.exc_value), unraisable.exc_value))
        path = _write_log("Unraisable error", tb, )
        # 不彈窗，僅記錄

    sys.unraisablehook = unraisable_hook

    # 3) Thread
    def th_hook(args: threading.ExceptHookArgs):
        exc = args.exc_value
        app._emit(f"Thread {args.thread.name} error", exc)
    threading.excepthook = th_hook

    # 4) asyncio
    try:
        loop = asyncio.get_event_loop()
        def aio_hook(loop, context):
            exc = context.get("exception") or RuntimeError(context.get("message"))
            app._emit("asyncio task error", exc)
        loop.set_exception_handler(aio_hook)
    except RuntimeError:
        pass

    # 5) Qt 訊息轉檔（非 Python 例外，如 qWarning）
    def qt_msg_handler(mode, ctx, msg):
        with open(os.path.join(LOG_DIR, "qt_messages.log"), "a", encoding="utf-8") as f:
            f.write(f"{_now()} [{mode}] {ctx.file}:{ctx.line} {msg}\n")
    qInstallMessageHandler(qt_msg_handler)

    # 6) 致命訊號回溯（C 擊穿時至少輸出 Python 堆疊）
    f = open(os.path.join(LOG_DIR, "fatal_signal.log"), "a", encoding="utf-8")
    faulthandler.enable(file=f, all_threads=True)

def connect_crash_dialog(app: CrashApp):
    def on_error(title: str, tb: str, e: Exception):
        sys.__excepthook__(type(e), e, e.__traceback__)
        path = _write_log(title, tb)
        box = Toast()
        box.setWindowTitle("App Crash")
        box.show_notice("fatal", "App Crash", f"{title}\n已寫入：\n{path}", ms=50000, allow_multiple=False, traceback=tb)
        box.exec()
        sys.exit(1)
    app.error_bus.error.connect(on_error)