# -*- coding: utf-8 -*-
"""
ui/threads.py — 背景移除整合版

提供三種引擎：
- rembg  (預設)
- openvino  (RMBG-1.4 或 U2Net 等 IR/ONNX，環境變數 RMBG_MODEL_PATH 指向模型檔)
- hsv   (純色背景移除，會自動估計色域，可調門檻)

並支援靜態圖片與動圖/影片（逐幀抽取→去背→合成 APNG/GIF/WEBM）。

相依：PyQt6, Pillow(PIL), numpy, opencv-python, (選用) openvino, rembg, ffmpeg
"""

import os
import sys
import io
import time
import tempfile
import shutil
import subprocess
from .ui_error import FFmpegNotFoundError
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from logging import Logger

import numpy as np
from PIL import Image

from PyQt6.QtCore import (
    QObject, QThread, pyqtSignal, pyqtSlot, QSize, QEventLoop, QTimer
)
from PyQt6.QtGui import QImageReader

# =============================
# 基礎資料結構
# =============================
@dataclass
class MediaInfo:
    path: str
    width: int
    height: int
    is_anim: bool
    bytes: int


# =============================
# 載入清單 Thread
# =============================
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
        self._exts = {".webp", ".gif"}

        self._thread: Optional[QThread] = None
        self._worker: Optional[_LoaderWorker] = None

    def set_directory(self, directory: str):
        self.directory = directory

    def load(self):
        self._start_worker()

    @pyqtSlot()
    def reload(self):
        self._start_worker()

    def _start_worker(self):
        # 防止對已刪除的 QThread 呼叫 isRunning()
        if self._thread is not None:
            try:
                if self._thread.isRunning():
                    return
            except RuntimeError:
                self._thread = None
        self._thread = QThread()
        self._worker = _LoaderWorker(self.directory, self._exts, self.logger)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.do_reload)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self.error)
        self._worker.error.connect(self.error_clear)

        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    @pyqtSlot(object)
    def error_clear(self, e):
        self._thread.quit()

    @pyqtSlot(dict)
    def _on_finished(self, payload: Dict):
        self.reload_finished.emit(payload)


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
# 去背工具 Thread（支援 rembg / openvino / hsv）

# HSV 參數（可調強度）
from dataclasses import dataclass

@dataclass
class HSVOpts:
    tol_h: int = 10        # 基本 Hue 容忍
    tol_s: int = 60        # 基本 Saturation 容忍
    tol_v: int = 60        # 基本 Value 容忍
    strength: float = 1.5  # 強度倍率（>1 更寬鬆、邊界更吃掉）
    erode_iter: int = 1    # 侵蝕次數（吃掉亮邊）
    dilate_iter: int = 0   # 膨脹次數
    feather_px: float = 2.0  # 距離轉換羽化半徑（像素）
    use_guided: bool = False  # 如有 ximgproc，可啟用 edge-aware 平滑
# =============================
class RmbgThread(QObject):
    """非阻塞去背：
        worker = RmbgThread(logger)
        worker.progress.connect(lambda p: ...)
        worker.finished.connect(lambda payload: ...)
        worker.error.connect(lambda e: ...)
        worker.remove_bg(src_path, out_dir='./animes', prefer='auto', engine='rembg')
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
        self._thread: Optional[QThread] = None
        self._worker: Optional[_RmbgWorker] = None

    def remove_bg(self, src_path: str, out_dir: str = "./animes",
                  prefer: str = "auto", engine: str = "hsv"):
        # 防止對已刪除的 QThread 呼叫 isRunning()
        if self._thread is not None:
            try:
                if self._thread.isRunning():
                    self.progress.emit({"signalId": "rmbg", "value": 100, "status": "loading model"})
                    return
            except RuntimeError:
                self._thread = None

        self._thread = QThread()
        self._worker = _RmbgWorker(src_path, out_dir, prefer, engine, self.logger)
        self._worker.moveToThread(self._thread)

        # wiring
        self._thread.started.connect(self._worker.do_remove)
        self._worker.progress.connect(self.progress)
        self._worker.finished.connect(self.finished)
        self._worker.error.connect(self.error)
        self._worker.error.connect(self.error_clear)
        self._worker.finished.connect(self._thread.quit)

        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    
    @pyqtSlot(object)
    def error_clear(self, e):
        self._thread.quit()
        # self._thread = None

class _RmbgWorker(QObject):
    progress = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(object)

    def __init__(self, src_path: str, out_dir: str, prefer: str, engine: str, logger: Logger):
        super().__init__()
        self.src_path = src_path
        self.out_dir = out_dir
        self.prefer = prefer
        self.engine = (engine or "rembg").lower()
        self.logger = logger
        # 延遲載入 OpenVINO 模型（需時）
        self._ov_compiled = None
        # HSV 可調參數（可依需求外部暴露）
        self.hsv_opts = HSVOpts()
        self.fps = 0
        # strength=1.5、erode_iter=1、feather_px=2.0
        # self.hsv_opts.strength = 0.15

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

    # 入口：依引擎去背單張，回傳 RGBA ndarray
    def _remove_image(self, src: str, out_dir: str) -> str:
        base = os.path.splitext(os.path.basename(src))[0]
        tmp_png = os.path.join(out_dir, f"{base}_rmbg_tmp.png")
        out_webp = os.path.join(out_dir, f"{base}_rmbg.webp")
        self.progress.emit({"signalId": "rmbg", "value": 5, "status": f"loading model ({self.engine})"})

        if self.engine == "openvino":
            rgba = self._remove_image_openvino(src)
        elif self.engine == "hsv":
            rgba = self._remove_image_hsv(src)
        else:
            rgba = self._remove_image_rembg(src)

        Image.fromarray(rgba, "RGBA").save(tmp_png)

        try:
            subprocess.run([
                "ffmpeg","-y","-i", tmp_png,
                "-c:v","libwebp_anim","-lossless","1",
                "-compression_level","9",
                "-preset","picture","-exact",  # 無損；改用 -q:v 調體積
                out_webp
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            os.remove(tmp_png)
            self.progress.emit({"signalId":"rmbg","value":100,"status":"done"})
            return out_webp
        except Exception:
            self.progress.emit({"signalId":"rmbg","value":100,"status":"done(png fallback)"})
        return tmp_png

        # Image.fromarray(rgba, mode="RGBA").save(out)
        # self.progress.emit({"signalId": "rmbg", "value": 100, "status": "done"})
        # return out

    # ========== 三種引擎 ==========
    def _remove_image_rembg(self, src: str) -> np.ndarray:
        try:
            from rembg import remove, new_session  # type: ignore
        except Exception as e:
            raise RuntimeError("rembg 未安裝，請先 `pip install rembg`。") from e
        sess = new_session("u2net")  # 你可改 isnet-general / birefnet-general / isnet-anime 等
        with open(src, "rb") as f:
            data = f.read()
        out_bytes = remove(data, session=sess)
        im = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
        return np.array(im)

    def _load_ov_model(self):
        if self._ov_compiled is not None:
            return self._ov_compiled
        model_path = os.getenv("RMBG_MODEL_PATH", "").strip()
        if not model_path or not os.path.exists(model_path):
            raise RuntimeError("未找到 RMBG 模型。請設定環境變數 RMBG_MODEL_PATH 指向 RMBG-1.4.onnx 或 IR .xml。")
        try:
            from openvino.runtime import Core
        except Exception as e:
            raise RuntimeError("需要 openvino。請先 `pip install openvino`。") from e
        core = Core()
        model = core.read_model(model_path)
        self._ov_compiled = core.compile_model(model, "AUTO")
        return self._ov_compiled

    def _remove_image_openvino(self, src: str) -> np.ndarray:
        import cv2
        compiled = self._load_ov_model()
        im = Image.open(src).convert("RGB")
        rgb = np.array(im)  # HWC uint8
        h, w = rgb.shape[:2]

        # 依常見 RMBG 入口尺寸（多為 1024）
        inp_size = 1024
        img_resized = cv2.resize(rgb, (inp_size, inp_size), interpolation=cv2.INTER_AREA)
        x = img_resized.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]  # NCHW

        req = compiled.create_infer_request()
        y = req.infer({compiled.inputs[0]: x})
        out_any = list(y.values())[0]
        mask = np.array(out_any).squeeze()  # HxW or 1xHxW

        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC)
        mask = np.clip(mask, 0.0, 1.0)
        alpha = (mask * 255).astype(np.uint8)
        rgba = np.dstack([rgb, alpha])
        return rgba

    def _estimate_bg_hsv_range(self, bgr: np.ndarray, pad=10, tol_h=10, tol_s=60, tol_v=60) -> Tuple[np.ndarray, np.ndarray]:
        """從影像四邊取樣，估計純色背景的 HSV 區間。
        - 強化通道健壯性，允許灰階/帶 alpha 的輸入。
        - 以 strength 倍率放寬門檻，避免白邊。
        - 扁平化後合併，避免尺寸不一致。"""
        import cv2
        # 標準化為 3 通道 BGR
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        elif bgr.shape[2] == 4:
            bgr = bgr[:, :, :3]

        H, W = bgr.shape[:2]
        p = max(1, min(pad, H // 2 if H > 1 else 1, W // 2 if W > 1 else 1))

        top    = bgr[:p, :, :].reshape(-1, 3)
        bottom = bgr[-p:, :, :].reshape(-1, 3)
        left   = bgr[:, :p, :].reshape(-1, 3)
        right  = bgr[:, -p:, :].reshape(-1, 3)
        border = np.concatenate([top, bottom, left, right], axis=0)
        if border.size == 0:
            border = bgr.reshape(-1, 3)

        hsv = cv2.cvtColor(border.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        h = int(np.median(hsv[:, 0]))
        s = int(np.median(hsv[:, 1]))
        v = int(np.median(hsv[:, 2]))

        k = float(self.hsv_opts.strength)
        th = int(round(tol_h * k))
        ts = int(round(tol_s * k))
        tv = int(round(tol_v * k))

        low  = np.array([max(0,   h - th), max(0,   s - ts), max(0,   v - tv)], dtype=np.uint8)
        high = np.array([min(179, h + th), min(255, s + ts), min(255, v + tv)], dtype=np.uint8)
        return low, high

    def _remove_image_hsv(self, src: str) -> np.ndarray:
        import cv2
        bgr = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        if bgr is None:
            raise RuntimeError("讀圖失敗")
        # 保證 3 通道 BGR
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        elif bgr.shape[2] == 4:
            bgr = bgr[:, :, :3]

        # HSV inRange（放寬門檻）
        low, high = self._estimate_bg_hsv_range(bgr, tol_h=self.hsv_opts.tol_h,
                                                tol_s=self.hsv_opts.tol_s,
                                                tol_v=self.hsv_opts.tol_v)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        bg = cv2.inRange(hsv, low, high)               # 背景=255

        # 強化：先膨脹背景再侵蝕前景，吃掉亮白邊
        if self.hsv_opts.dilate_iter > 0:
            k = np.ones((3, 3), np.uint8)
            bg = cv2.dilate(bg, k, iterations=self.hsv_opts.dilate_iter)
        if self.hsv_opts.erode_iter > 0:
            k = np.ones((3, 3), np.uint8)
            bg = cv2.erode(bg, k, iterations=self.hsv_opts.erode_iter)

        # 生成前景二值與距離羽化 alpha
        fg_bin = cv2.bitwise_not(bg)                   # 前景=255
        if self.hsv_opts.feather_px > 0:
            # 距離轉換 -> 線性羽化到指定像素
            dist = cv2.distanceTransform(fg_bin, cv2.DIST_L2, 3)
            soft = np.clip(dist / float(self.hsv_opts.feather_px), 0.0, 1.0)
            alpha = (soft * 255.0).astype(np.uint8)
        else:
            alpha = fg_bin

        # 可選：guided/dt 濾波，讓 alpha 貼邊（需 ximgproc）
        if self.hsv_opts.use_guided:
            try:
                import cv2.ximgproc as xip
                guide = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                alpha = xip.guidedFilter(guide, alpha, radius=4, eps=1e-3)
            except Exception:
                pass

        rgba = np.dstack([bgr[..., ::-1], alpha])      # BGR->RGB + alpha
        return rgba

    # ========== 動圖/影片處理 ==========
    def _remove_anim_or_video(self, src: str, out_dir: str, prefer: str) -> dict:
        if not self._check_ffmpeg():
            raise FFmpegNotFoundError("需要 ffmpeg 以處理動圖/影片。請安裝並加入 PATH。")

        base = os.path.splitext(os.path.basename(src))[0]
        tmp = tempfile.mkdtemp(prefix="rmbg_")
        frames_dir = os.path.join(tmp, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # 抽幀
        self.progress.emit({"signalId": "rmbg", "value": 5, "status": "extract frames"})
        extract_args = [
            "ffmpeg", "-y", "-i", src,"-vsync","0",
            os.path.join(frames_dir, "f_%06d.png")
        ]
        subprocess.run(extract_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(".png")])
        total = max(1, len(files))

        # 逐幀去背
        out_frames = os.path.join(tmp, "out")
        os.makedirs(out_frames, exist_ok=True)
        for i, fname in enumerate(files, 1):
            fpath = os.path.join(frames_dir, fname)
            if self.engine == "openvino":
                rgba = self._remove_image_openvino(fpath)
            elif self.engine == "hsv":
                rgba = self._remove_image_hsv(fpath)
            else:
                rgba = self._remove_image_rembg(fpath)
            Image.fromarray(rgba, "RGBA").save(os.path.join(out_frames, fname))
            if i % 3 == 0 or i == total:
                pct = 5 + int(80 * i / total)
                self.progress.emit({"signalId": "rmbg", "value": pct, "status": f"frame {i}/{total}"})

        # 合成：優先 APNG，其次 GIF，再者 WEBM（透明）
        prefer = (prefer or "auto").lower()
        out_path: Optional[str] = None

        # 取得 fps（失敗則 15）
        def _probe_fps(p):
            try:
                r = subprocess.run(
                    ["ffprobe","-v","error","-select_streams","v:0",
                    "-show_entries","stream=avg_frame_rate",
                    "-of","default=nw=1:nk=1", p],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
                num,den = r.stdout.strip().split("/")
                num,den = int(num), (int(den) if den!="0" else 1)
                return max(1, round(num/den))
            except Exception:
                return 15
            
        fps = _probe_fps(src)
        self.fps = fps

        self.progress.emit({"signalId": "rmbg", "value": 90, "status": "encode"})

        # "-c:v","libwebp_anim","-lossless","1",
        #         "-compression_level","9",
        #         "-preset","picture","-exact",

        out_webp = os.path.join(out_dir, f"{base}_rmbg547.webp")
        subprocess.run([
        "ffmpeg","-y","-framerate", str(fps),
        "-i", os.path.join(out_frames,"f_%06d.png"),
        "-c:v","libwebp_anim",
        "-pix_fmt","yuva420p",   # 保透明
        "-loop","0",            # 無限循環
        "-q:v","75",             # 或改 -lossless 1
        out_webp
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # if not out_path:
        #     # 退而求其次：輸出去背後的逐幀資料夾
        #     out_path = os.path.join(out_dir, f"{base}_rmbg_frames")
        #     if os.path.exists(out_path):
        #         shutil.rmtree(out_path)
        #     shutil.copytree(out_frames, out_path)

        # 清理暫存
        shutil.rmtree(tmp, ignore_errors=True)
        # os.remove(f"./animes/{base}_rmbg_frames",)
        self.progress.emit({"signalId": "rmbg", "value": 100, "status": "done"})
        return {"input": src, "output": out_path, "kind": "anim", "frames": len(files)}
    
# def _test_rm(path):
#     n = 0
#     tmp = tempfile.mkdtemp(prefix="rmbg_")
#     frames_dir = os.path.join(tmp, "frames")
#     os.makedirs(frames_dir, exist_ok=True)
#     print(frames_dir)
#     extract_args = [
#             "ffmpeg", "-y", "-i", path,"-vsync","0",
#             os.path.join(frames_dir, "f_%06d.png")
#         ]
#     subprocess.run(extract_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
#     source = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(".png")])
#       # source 為要轉存的所有圖片陣列 ( opencv 格式，色彩為 RGBA )
#     # for i in source:                  # source 為要轉存的所有圖片陣列 ( opencv 格式，色彩為 RGBA )
#     #     img = Image.open(frames_dir + '\\' +i)      # 轉換成 PIL 格式
#     #     img.save(f'./gif{n}.gif')  # 儲存為 gif
#     #     n = n + 1                     # 改變儲存的檔名編號

#     output = []                       # 建立空串列
#     for i in source:
#         img = Image.open(frames_dir + '\\' + i)      # 轉換成 PIL 格式
#         img = img.convert("RGBA")             # 轉換為 RGBA
#         output.append(img)                    # 記錄每張圖片內容

#     # 轉存為 gif 動畫，設定 disposal=2
#     shutil.rmtree(tmp, ignore_errors=True)
#     output[0].save("oxxostudio.gif", save_all=True, append_images=output[1:], duration=100, loop=0, disposal=2)
# if __name__ == "__main__":
#     _test_rm(r'C:\Users\car\Downloads\lixovsk-hoshimi-miyabi.gif')
