# src/loger.py
import logging
import os
from datetime import datetime

#####
# 🔥📌✅📂📄🔹🎨😃🚀❌🔧⚠📁🗑️🔇🛑🎙️🎤

C = {
    "red": "\033[91m",
    "green": "\033[92m",
    "blue": "\033[94m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
    "magenta": "\033[95m",
    "black": "\033[30m",
    "white": "\033[97m",
    "gray": "\033[37m",
    "orange": "\033[33m",
    "purple": "\033[35m",
    "pink": "\033[95m",
    # 背景色
    "bg_red": "\033[41m",
    "bg_green": "\033[42m",
    "bg_blue": "\033[44m",
    "bg_yellow": "\033[43m",
    "bg_cyan": "\033[46m",
    "bg_magenta": "\033[45m",
    "bg_black": "\033[40m",
    "bg_white": "\033[47m",
    "r": "\033[0m",
}


class levelFormatter(logging.Formatter):
    """讓不同 Log Level 使用不同格式"""

    def __init__(self, fmt_dict):
        super().__init__()
        self.fmt_dict = fmt_dict
        self.default_fmt = fmt_dict.get(logging.DEBUG, "%(levelname)s: %(message)s")

    def format(self, record):
        log_fmt = self.fmt_dict.get(record.levelno, self.default_fmt)
        formatter = logging.Formatter(
            log_fmt,
        )
        return formatter.format(record)


class loggerFactory:
    def __init__(
        self,
        logger_name: str = "main",
        log_level: int | str = logging.DEBUG,
        write_log: bool = False,
        file_name: str = "log",
        path: str = "./logs",
    ):
        if isinstance(log_level, str):
            log_level = logging._nameToLevel.get(log_level.upper(), logging.INFO)

        self.write_log = write_log
        log_path = os.path.join(os.getcwd(), path)
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        fname = datetime.now().strftime("%y-%m-%d-%H%M%S") + file_name + ".log"

        log_file = os.path.join(log_path, fname)

        self.logger = logging.getLogger(
            logger_name,
        )
        self.logger.setLevel(log_level)
        self.logger.propagate = False  # 防止重複打印

        if not self.logger.handlers:
            # 設定 Log 格式
            log_formats = {
                logging.DEBUG: f"%(asctime)s [%(name)s] {C['cyan']}[DEBUG] | %(message)s"
                + C["r"],
                logging.INFO: f"%(asctime)s [%(name)s] {C['green']}[INFO]{C['r']}  | %(message)s"
                + C["r"],
                logging.WARNING: f"%(asctime)s [%(name)s] {C['yellow']}[WARN]  | %(message)s"
                + C["r"],
                logging.ERROR: f"%(asctime)s [%(name)s] {C['red']}[ERROR] | %(message)s"
                + C["r"],
                logging.CRITICAL: f"%(asctime)s [%(name)s] ❌{C['bg_red'] + C['white']}[CRITICAL] | %(message)s"
                + C["r"]
                + "❌",
            }
            formatter = levelFormatter(log_formats)

            # Console Handler (顯示在終端機)
            # stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            # 加入 Handler
            if self.write_log:
                print(f"Log file: {log_file}")
                # File Handler (寫入檔案)
                file_handler = logging.FileHandler(log_file, encoding="utf-8")
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def getLogger(self):
        return self.logger


class testlog(loggerFactory):
    def __init__(self):
        super().__init__(write_log=True)
        self.logger = self.getLogger()
        self.logger.info("test✅✅")

    def test(self):
        self.logger.debug("debug")
        self.logger.info("info✅✅")
        self.logger.warning("warning")
        self.logger.error("error")
        self.logger.critical("critical")


if __name__ == "__main__":
    test = testlog()
    test.test()
