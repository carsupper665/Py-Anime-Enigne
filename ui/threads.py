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
DEFAULTS = {
    "border": 12,            # 取四邊厚度作背景取樣
    "kmeans_k": 2,           # 邊框分群以避開陰影
    "use_mahalanobis": True, # True=用馬氏距離，False=用歐氏距離
    "morph_open_close": True,
    "keep_largest": True,    # 只留最大前景
    "min_keep_area": 400,
    "near_radius_ratio": 0.35,
    "feather_px": 2.5
}
import cv2
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
def _estimate_bg_lab_from_borders(lab, border=10, k=2):
    h, w = lab.shape[:2]
    S = np.vstack([
        lab[0:border, :, :].reshape(-1,3),
        lab[h-border:h, :, :].reshape(-1,3),
        lab[:, 0:border, :].reshape(-1,3),
        lab[:, w-border:w, :].reshape(-1,3)
    ]).astype(np.float32)

    if k >= 2 and S.shape[0] >= k:
        criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _ret, labels, centers = cv2.kmeans(S, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        bg = centers[np.bincount(labels.ravel()).argmax()]
        cluster = S[labels.ravel()==np.bincount(labels.ravel()).argmax()]
    else:
        bg = np.median(S, axis=0)
        cluster = S

    cov = np.cov(cluster.T) + np.eye(3)*1e-6   # 防奇異矩陣
    inv_cov = np.linalg.inv(cov)
    mean = bg.astype(np.float32)
    return mean, inv_cov

def _bg_mask_by_color_and_border(bgr, opts):
    # BGR -> Lab（用 float64，後續距離與逆協方差一致）
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)  # cvtColor 說明見官方文件。 
    mean, inv_cov = _estimate_bg_lab_from_borders(
        lab, border=int(opts["border"]), k=int(opts["kmeans_k"])
    )  # 回傳的 mean, inv_cov 已基於邊框樣本計算

    H, W = lab.shape[:2]
    X = lab.reshape(-1, 3)  # (N,3) float64

    # 距離圖：Mahalanobis 或 Euclidean（二擇一）
    if opts.get("use_mahalanobis", True):
        # 向量化馬氏距離，避免 cv2.Mahalanobis 的 dtype 斷言
        diff = X - mean.astype(np.float64)                   # (N,3)
        inv_cov = inv_cov.astype(np.float64)                 # (3,3)
        d = np.sqrt(np.einsum('ij,jk,ik->i', diff, inv_cov, diff))  # (N,)
    else:
        d = np.linalg.norm(X - mean.astype(np.float64), axis=1)

    dist = d.reshape(H, W)

    # Otsu 取閾：距離小=背景
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _thr, mask_bg0 = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 只保留「接觸影像邊界」的連通塊為背景
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask_bg0 > 0).astype(np.uint8), connectivity=4)
    touch = np.zeros(num, dtype=bool)
    if num > 1:
        touch[labels[0, :]] = True
        touch[labels[-1, :]] = True
        touch[labels[:, 0]] = True
        touch[labels[:, -1]] = True

    bg = np.zeros_like(mask_bg0)
    for i in range(1, num):
        if touch[i]:
            bg[labels == i] = 255
    return bg

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
                  prefer: str = "auto", engine: str = "hsv",
                  hsv_cfg: dict | None = None, wand: dict | None = None,
                  image_format: str | None = None, anim_format: str | None = None,
                  range_ms: tuple[int | None, int | None] | None = None): # rembg / openvino / hsv / wand
        # 防止對已刪除的 QThread 呼叫 isRunning()
        if self._thread is not None:
            try:
                if self._thread.isRunning():
                    self.progress.emit({"signalId": "rmbg", "value": 100, "status": "loading model"})
                    return
            except RuntimeError:
                self._thread = None

        self._thread = QThread()
        self._worker = _RmbgWorker(src_path, out_dir, prefer, engine, self.logger,
                                   hsv_cfg=hsv_cfg or {}, wand_cfg=wand or {},
                                   image_format=(image_format or 'webp'),
                                   anim_format=(anim_format or 'webp'),
                                   range_ms=range_ms)
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

    def cancel_current(self):
        try:
            if self._worker is not None:
                self._worker.request_cancel()
        except Exception:
            pass

class _RmbgWorker(QObject):
    progress = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(object)

    def __init__(self, src_path: str, out_dir: str, prefer: str, engine: str, logger: Logger,
                 hsv_cfg: dict, wand_cfg: dict, image_format: str, anim_format: str,
                 range_ms: tuple[int | None, int | None] | None = None):
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
        # 由設定注入
        try:
            self.hsv_opts.tol_h = int(hsv_cfg.get("tol_h", self.hsv_opts.tol_h))
            self.hsv_opts.tol_s = int(hsv_cfg.get("tol_s", self.hsv_opts.tol_s))
            self.hsv_opts.tol_v = int(hsv_cfg.get("tol_v", self.hsv_opts.tol_v))
            self.hsv_opts.strength = float(hsv_cfg.get("strength", self.hsv_opts.strength))
            self.hsv_opts.erode_iter = int(hsv_cfg.get("erode_iter", self.hsv_opts.erode_iter))
            self.hsv_opts.dilate_iter = int(hsv_cfg.get("dilate_iter", self.hsv_opts.dilate_iter))
            self.hsv_opts.feather_px = float(hsv_cfg.get("feather_px", self.hsv_opts.feather_px))
            self.hsv_opts.use_guided = bool(hsv_cfg.get("use_guided", self.hsv_opts.use_guided))
        except Exception:
            pass
        # 魔術棒參數
        self.wand_opts = wand_cfg or {}
        self.wand_seed = None
        if isinstance(self.wand_opts.get("seed", None), (tuple, list)) and len(self.wand_opts["seed"]) == 2:
            self.wand_seed = (int(self.wand_opts["seed"][0]), int(self.wand_opts["seed"][1]))
        # 輸出格式
        self.image_format = (image_format or 'webp').lower()
        # 規格：影片/動圖輸出強制 animated-webp
        self.anim_format = 'webp'
        self.fps = 0
        self.range_ms = range_ms or (None, None)
        # strength=1.5、erode_iter=1、feather_px=2.0
        self.hsv_opts.strength = 1.5
        self._cancel_requested = False  # OpenSpec: add-processing-queue — cancel current job
        # spec: openspec/changes/add-processing-queue/tasks.md:1

    def request_cancel(self):
        self._cancel_requested = True

    def _is_canceled(self) -> bool:
        return bool(self._cancel_requested)

    @pyqtSlot()
    def do_remove(self):
        try:
            if self._is_canceled():
                raise RuntimeError("canceled")
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

    # 靜態圖輸出
    def _remove_image(self, src: str, out_dir: str) -> str:
        if self._is_canceled():
            raise RuntimeError("canceled")
        base = os.path.splitext(os.path.basename(src))[0]
        tmp_png = os.path.join(out_dir, f"{base}_rmbg_tmp.png")
        out_png = os.path.join(out_dir, f"{base}_rmbg.png")
        out_webp = os.path.join(out_dir, f"{base}_rmbg.webp")
        self.progress.emit({"signalId": "rmbg", "value": 5, "status": f"loading model ({self.engine})"})

        if self._is_canceled():
            raise RuntimeError("canceled")
        if self.engine == "openvino":
            rgba = self._remove_image_openvino(src)
        elif self.engine == "hsv":
            rgba = self._remove_image_hsv(src)
        elif self.engine == "wand":
            if not self.wand_seed:
                raise RuntimeError("魔術棒需要先在圖片上取樣（點擊）座標。")
            rgba = self._remove_image_magicwand(src, self.wand_seed, self.wand_opts)
        else:
            rgba = self._remove_image_rembg(src)

        Image.fromarray(rgba, "RGBA").save(tmp_png)

        fmt = (self.image_format or 'webp').lower()
        if fmt == 'png':
            # 直接輸出 PNG（無損）
            try:
                os.replace(tmp_png, out_png)
            except Exception:
                shutil.copyfile(tmp_png, out_png)
                os.remove(tmp_png)
            self.progress.emit({"signalId":"rmbg","value":100,"status":"done(png)"})
            return out_png
        else:
            # WEBP（使用 ffmpeg 確保一致性）
            try:
                subprocess.run([
                    "ffmpeg","-y","-i", tmp_png,
                    "-c:v","libwebp","-lossless","1",
                    "-compression_level","6",
                    "-preset","picture",
                    out_webp
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                os.remove(tmp_png)
                self.progress.emit({"signalId":"rmbg","value":100,"status":"done(webp)"})
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
            if os.path.exists("./models/model.onnx"):
                model_path = "./models/model.onnx"
            else:
                raise FFmpegNotFoundError("未找到 RMBG 模型。請設定環境變數 RMBG_MODEL_PATH 指向 RMBG-1.4.onnx 或 IR .xml。")
        try:
            from openvino.runtime import Core
        except Exception as e:
            raise RuntimeError("需要 openvino。請先 `pip install openvino`。") from e
        core = Core()
        model = core.read_model(model_path)
        self._ov_compiled = core.compile_model(model, "AUTO")
        return self._ov_compiled

    def _remove_image_openvino(self, src: str) -> np.ndarray:
        
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


    # --- 修改: 去背，接受 seed_xy；None 則退回四角 ---
    def _remove_image_magicwand(self, src: str, seed_xy: tuple[int,int], opts: Optional[dict] = None) -> np.ndarray:
        from core.wand import compute_mask
        bgr = cv2.imread(src, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("讀圖失敗")
        mask = compute_mask(bgr, seed_xy, opts or {})
        rgb = bgr[..., ::-1]
        rgba = np.dstack([rgb, mask])
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
        # in/out（毫秒）
        t_in, t_out = self.range_ms
        extract_args = ["ffmpeg", "-y"]
        if isinstance(t_in, int) and t_in > 0:
            extract_args += ["-ss", f"{t_in/1000:.3f}"]
        extract_args += ["-i", src, "-vsync", "0"]
        if isinstance(t_out, int) and t_out > 0:
            # 使用 -to 終點時間（以輸入時間軸為準）
            extract_args += ["-to", f"{t_out/1000:.3f}"]
        extract_args += [os.path.join(frames_dir, "f_%06d.png")]
        try:
            subprocess.run(extract_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

        files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(".png")])
        total = max(1, len(files))

        # 逐幀去背
        out_frames = os.path.join(tmp, "out")
        os.makedirs(out_frames, exist_ok=True)
        try:
            for i, fname in enumerate(files, 1):
                if self._is_canceled():
                    raise RuntimeError("canceled")
                fpath = os.path.join(frames_dir, fname)
                if self.engine == "openvino":
                    rgba = self._remove_image_openvino(fpath)
                elif self.engine == "hsv":
                    rgba = self._remove_image_hsv(fpath)
                elif self.engine == "wand":
                    if not self.wand_seed:
                        raise RuntimeError("魔術棒需要先在圖片上取樣（點擊）座標。")
                    rgba = self._remove_image_magicwand(fpath, self.wand_seed, self.wand_opts)
                else:
                    rgba = self._remove_image_rembg(fpath)
                Image.fromarray(rgba, "RGBA").save(os.path.join(out_frames, fname))
                if i % 3 == 0 or i == total:
                    pct = 5 + int(80 * i / total)
                    self.progress.emit({"signalId": "rmbg", "value": pct, "status": f"frame {i}/{total}"})
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

        # 合成輸出：依設定 anim_format 選擇 webp 或 gif
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
        # 規格：固定輸出 animated-webp
        out_webp = os.path.join(out_dir, f"{base}_rmbg.webp")
        try:
            subprocess.run([
            "ffmpeg","-y","-framerate", str(fps),
            "-i", os.path.join(out_frames,"f_%06d.png"),
            "-c:v","libwebp_anim",
            "-pix_fmt","yuva420p",   # 保透明
            "-loop","0",            # 無限循環
            "-q:v","75",
            out_webp
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            out_path = out_webp
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

        # if not out_path:
        #     # 退而求其次：輸出去背後的逐幀資料夾
        #     out_path = os.path.join(out_dir, f"{base}_rmbg_frames")
        #     if os.path.exists(out_path):
        #         shutil.rmtree(out_path)
        #     shutil.copytree(out_frames, out_path)

        # 清理暫存
        shutil.rmtree(tmp, ignore_errors=True)
        self.progress.emit({"signalId": "rmbg", "value": 100, "status": "done"})
        return {"input": src, "output": out_path, "kind": "anim", "frames": len(files)}
    
