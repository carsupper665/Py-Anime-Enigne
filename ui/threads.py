# ./ui/threads.py
import os
import sys
import time
import tempfile
import shutil
import subprocess
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from logging import Logger

from PyQt6.QtCore import (
    QObject, QThread, pyqtSignal, pyqtSlot, QSize, QEventLoop, QTimer
)
from PyQt6.QtGui import QImageReader

@dataclass
class MediaInfo:
    path: str
    width: int
    height: int
    is_anim: bool
    bytes: int
    
class LoadThread(QObject):
    progress = pyqtSignal(dict)
    finished = pyqtSignal(dict)

    def __init__(self, logger: Logger):
        super().__init__()
        self.logger = logger

    def load(self):
        v = 0
        self.emit_helper("load", v, "Loading anime data...")
        self.loader = GifLoader(self.logger)
        payload = {}
        loop = QEventLoop()

        def on_done(p):
            nonlocal payload
            payload.update(p)
            loop.quit()

        def on_error(e):
            self.progress.emit({"signalId": "load", "value": v, "status": f"error: {e}"})
            loop.quit()

        self.loader.reload_finished.connect(on_done)
        self.loader.error.connect(on_error)
        self.loader.load()

        v += 10
        QTimer.singleShot(0, lambda: self.emit_helper("load", v, "Almost done..."))
        loop.exec()

        for i in range(v, 101):
            time.sleep(0.01)
            self.emit_helper("load", i, "Welcome!")
        self.finished.emit(payload)

    def emit_helper(self, id: str, value: int, status: str):
        self.progress.emit({"signalId": id, "value": value, "status": status})


class GifLoader(QObject):
    reload_finished = pyqtSignal(dict)
    error = pyqtSignal(object)

    def __init__(self, logger: Logger, directory: str = "./animes"):
        super().__init__()
        self.logger = logger
        self.directory = directory
        self._exts = {".gif", ".webp", ".png", ".jpg", ".jpeg", ".apng"}

        self._thread: QThread | None = None
        self._worker: _LoaderWorker | None = None

    def set_directory(self, directory: str):
        self.directory = directory

    def load(self):
        self._start_worker()

    @pyqtSlot()
    def reload(self):
        self._start_worker()

    def _start_worker(self):
        if self._thread and self._thread.isRunning():
            return  # 忽略同時重入
        self._thread = QThread()
        self._worker = _LoaderWorker(self.directory, self._exts, self.logger)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.do_reload)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self.error)

        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    @pyqtSlot(dict)
    def _on_finished(self, payload: Dict):
        self.reload_finished.emit(payload)
        # _thread.quit() 已在啟動處連接


class _LoaderWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(object)

    def __init__(self, directory: str, exts: set[str], logger: Logger):
        super().__init__()
        self.dir = directory
        self.exts = exts
        self.logger = logger

    @pyqtSlot()
    def do_reload(self):
        try:
            items: List[Dict] = []
            if not os.path.isdir(self.dir):
                self.finished.emit({"dir": self.dir, "items": items})
                return

            for root, _, files in os.walk(self.dir):
                for name in files:
                    path = os.path.join(root, name)
                    ext = os.path.splitext(name)[1].lower()
                    if ext not in self.exts:
                        continue

                    reader = QImageReader(path)
                    if not reader.canRead():
                        continue
                    size: QSize = reader.size()
                    is_anim: bool = reader.supportsAnimation()

                    info = MediaInfo(
                        path=path,
                        width=size.width(),
                        height=size.height(),
                        is_anim=is_anim,
                        bytes=os.path.getsize(path) if os.path.exists(path) else 0,
                    )
                    items.append(asdict(info))

            self.finished.emit({"dir": self.dir, "items": items})
        except Exception as e:
            self.error.emit(e)


# =============================
# 去背工具（新增）
# =============================
class RmbgThread(QObject):
    """前台可呼叫：以非阻塞方式進行去背。
    用法：
        worker = RmbgThread(logger)
        worker.progress.connect(lambda p: ...)
        worker.finished.connect(lambda payload: ...)
        worker.error.connect(lambda e: ...)
        worker.remove_bg(src_path, out_dir='./animes', prefer='auto')
    """

    progress = pyqtSignal(dict)  # {signalId, value, status}
    finished = pyqtSignal(dict)  # {input, output, kind, frames?, manifest?}
    error = pyqtSignal(object)

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
    ANIM_EXTS = {".gif", ".webp", ".apng"}
    VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}

    def __init__(self, logger: Logger):
        super().__init__()
        self.logger = logger
        self._thread: QThread | None = None
        self._worker: _RmbgWorker | None = None

    def remove_bg(self, src_path: str, out_dir: str = "./animes", prefer: str = "auto"):
        """prefer: 'auto' | 'gif' | 'apng' | 'webm' | 'png'（靜態）"""
        if self._thread and self._thread.isRunning():
            return
        self._thread = QThread()
        self._worker = _RmbgWorker(src_path, out_dir, prefer, self.logger)
        self._worker.moveToThread(self._thread)

        # wiring
        self._thread.started.connect(self._worker.do_remove)
        self._worker.progress.connect(self.progress)
        self._worker.finished.connect(self.finished)
        self._worker.error.connect(self.error)
        self._worker.finished.connect(self._thread.quit)

        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()


class _RmbgWorker(QObject):
    progress = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(object)

    def __init__(self, src_path: str, out_dir: str, prefer: str, logger: Logger):
        super().__init__()
        self.src_path = src_path
        self.out_dir = out_dir
        self.prefer = prefer
        self.logger = logger

    @pyqtSlot()
    def do_remove(self):
        try:
            if not os.path.isfile(self.src_path):
                raise FileNotFoundError(self.src_path)
            os.makedirs(self.out_dir, exist_ok=True)

            ext = os.path.splitext(self.src_path)[1].lower()
            if ext in RmbgThread.IMAGE_EXTS and ext not in RmbgThread.ANIM_EXTS:
                out = self._remove_image(self.src_path, self.out_dir)
                self.finished.emit({"input": self.src_path, "output": out, "kind": "image"})
                return

            if ext in RmbgThread.ANIM_EXTS or ext in RmbgThread.VIDEO_EXTS:
                payload = self._remove_anim_or_video(self.src_path, self.out_dir, self.prefer)
                self.finished.emit(payload)
                return

            # 其他格式一律當作圖片處理
            out = self._remove_image(self.src_path, self.out_dir)
            self.finished.emit({"input": self.src_path, "output": out, "kind": "image"})
        except Exception as e:
            self.error.emit(e)

    # ---- helpers ----
    def _check_ffmpeg(self) -> bool:
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True
        except Exception:
            return False

    def _import_rembg(self):
        try:
            from rembg import remove, new_session  # type: ignore
            return remove, new_session
        except Exception as e:
            raise RuntimeError("rembg 未安裝，請先 `pip install rembg` 或於環境中提供。") from e

    def _remove_image(self, src: str, out_dir: str) -> str:
        remove, new_session = self._import_rembg()
        base = os.path.splitext(os.path.basename(src))[0]
        out = os.path.join(out_dir, f"{base}_rmbg.png")
        self.progress.emit({"signalId": "rmbg", "value": 5, "status": "loading model"})
        sess = new_session("u2net")  # 可改其他模型
        with open(src, "rb") as f:
            data = f.read()
        self.progress.emit({"signalId": "rmbg", "value": 20, "status": "processing"})
        result = remove(data, session=sess)
        with open(out, "wb") as f:
            f.write(result)
        self.progress.emit({"signalId": "rmbg", "value": 100, "status": "done"})
        return out

    def _remove_anim_or_video(self, src: str, out_dir: str, prefer: str) -> dict:
        if not self._check_ffmpeg():
            raise RuntimeError("需要 ffmpeg 以處理動圖/影片。請安裝並加入 PATH。")

        remove, new_session = self._import_rembg()
        sess = new_session("u2net")

        base = os.path.splitext(os.path.basename(src))[0]
        tmp = tempfile.mkdtemp(prefix="rmbg_")
        frames_dir = os.path.join(tmp, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # 抽幀
        self.progress.emit({"signalId": "rmbg", "value": 5, "status": "extract frames"})
        # 使用 png 保留 alpha
        extract_args = [
            "ffmpeg", "-y", "-i", src,
            os.path.join(frames_dir, "f_%06d.png")
        ]
        subprocess.run(extract_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # 清單
        files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(".png")])
        total = max(1, len(files))

        # 逐幀去背
        out_frames = os.path.join(tmp, "out")
        os.makedirs(out_frames, exist_ok=True)
        for i, fname in enumerate(files, 1):
            fpath = os.path.join(frames_dir, fname)
            with open(fpath, "rb") as fr:
                buf = fr.read()
            out_buf = remove(buf, session=sess)
            with open(os.path.join(out_frames, fname), "wb") as fw:
                fw.write(out_buf)
            if i % 3 == 0 or i == total:
                pct = 5 + int(80 * i / total)
                self.progress.emit({"signalId": "rmbg", "value": pct, "status": f"frame {i}/{total}"})

        # 重組：優先 APNG，其次 GIF，再者 WEBM 透明
        prefer = prefer or "auto"
        out_path: Optional[str] = None

        def try_make_apng() -> Optional[str]:
            """使用 ffmpeg 合成 APNG（需要支援 apng muxer 的 ffmpeg）。"""
            try:
                out = os.path.join(out_dir, f"{base}_rmbg.apng")
                args = [
                    "ffmpeg", "-y", "-framerate", "15", "-i", os.path.join(out_frames, "f_%06d.png"),
                    "-plays", "0", out
                ]
                subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                return out
            except Exception:
                return None

        def try_make_gif() -> Optional[str]:
            # GIF 僅 1-bit 透明，邊緣可能鋸齒
            try:
                import imageio.v2 as imageio
                import numpy as np
                outs = [imageio.imread(os.path.join(out_frames, f)) for f in files]
                out = os.path.join(out_dir, f"{base}_rmbg.gif")
                imageio.mimsave(out, outs, format="GIF", duration=1/15)
                return out
            except Exception:
                return None

        def try_make_webm() -> Optional[str]:
            try:
                out = os.path.join(out_dir, f"{base}_rmbg.webm")
                args = [
                    "ffmpeg", "-y", "-framerate", "15", "-i", os.path.join(out_frames, "f_%06d.png"),
                    "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-auto-alt-ref", "0", out
                ]
                subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                return out
            except Exception:
                return None

        self.progress.emit({"signalId": "rmbg", "value": 90, "status": "encode"})

        if prefer == "apng" or prefer == "auto":
            out_path = try_make_apng()
        if not out_path and (prefer == "gif" or prefer == "auto"):
            out_path = try_make_gif()
        if not out_path and (prefer in ("webm", "auto")):
            out_path = try_make_webm()

        if not out_path:
            # 退而求其次：輸出去背後的逐幀資料夾
            out_path = os.path.join(out_dir, f"{base}_rmbg_frames")
            if os.path.exists(out_path):
                shutil.rmtree(out_path)
            shutil.copytree(out_frames, out_path)

        # 清理暫存
        shutil.rmtree(tmp, ignore_errors=True)
        self.progress.emit({"signalId": "rmbg", "value": 100, "status": "done"})
        return {"input": src, "output": out_path, "kind": "anim", "frames": len(files)}