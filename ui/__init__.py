# ui/__init__.py
from .nav_bar import NavBar
from .loading_page import LoadingPage
from .toast import Toast, LoadingToast
from .home_page import HomePage
from .logger import loggerFactory, C
from .ui_error import UnexpectedError, connect_crash_dialog, MainApp, install_global_handlers
from .threads import *
from .queue import ProcessingQueue, QueueJob
from .add_page import AddPage
from .edit_page import EditPage
from .settings_page import SettingsPage
from .preview_dialog import PreviewImageDialog

INFO = "info"
WARN = "warn"
ERROR = "error"
DEBUG = "debug"
FATAL = "fatal"
COLOR = C

__all__ = ["NavBar", "LoadingPage", "HomePage", "Toast", 
           "UnexpectedError", "INFO", "WARN", "ERROR", 
           "DEBUG", "FATAL", "loggerFactory", "LoadThread",
           "AddPage", "COLOR", "GifLoader", "MediaInfo", 
           "EditPage", "connect_crash_dialog", "MainApp",
            "install_global_handlers", "RmbgThread",
            "LoadingToast", "FFmpegNotFoundError", "SettingsPage", "ProcessingQueue", "QueueJob", "PreviewImageDialog"
           ]
