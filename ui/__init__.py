# ui/__init__.py
from .nav_bar import NavBar
from .loading_page import LoadingPage
from .toast import Toast
from .home_page import HomePage
from .logger import loggerFactory, C
from .ui_error import UnexpectedError, connect_crash_dialog, MainApp, install_global_handlers
from .threads import *
from .add_page import AddPage
from .edit_page import EditPage

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
            "install_global_handlers", "RmbgThread"
           ]

