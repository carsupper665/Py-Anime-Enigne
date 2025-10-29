# ui/__init__.py
from .nav_bar import NavBar
from .loading_page import LoadingPage
from .toast import Toast, LoadingToast
from .home_page import HomePage
from .logger import loggerFactory, C
from .ui_error import UnexpectedError, connect_crash_dialog, MainApp, install_global_handlers
try:
    from .threads import *
    _THREADS_AVAILABLE = True
except ImportError:
    _THREADS_AVAILABLE = False
from .queue import ProcessingQueue, QueueJob
from .add_page import AddPage
from .edit_page import EditPage
from .settings_page import SettingsPage
from .preview_dialog import PreviewImageDialog
from .osd import OSD

INFO = "info"
WARN = "warn"
ERROR = "error"
DEBUG = "debug"
FATAL = "fatal"
COLOR = C

__all__ = [
    "NavBar",
    "LoadingPage",
    "HomePage",
    "Toast",
    "UnexpectedError",
    "INFO",
    "WARN",
    "ERROR",
    "DEBUG",
    "FATAL",
    "loggerFactory",
    "AddPage",
    "COLOR",
    "EditPage",
    "connect_crash_dialog",
    "MainApp",
    "install_global_handlers",
    "LoadingToast",
    "FFmpegNotFoundError",
    "SettingsPage",
    "ProcessingQueue",
    "QueueJob",
    "PreviewImageDialog",
    "OSD",
]

if _THREADS_AVAILABLE:
    __all__.extend(["LoadThread", "GifLoader", "MediaInfo", "RmbgThread"])
