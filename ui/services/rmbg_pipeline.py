from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import logging
import numpy as np
from PIL import Image

from core.diagnostics import attach_diagnostic, log_structured
from ui.ui_error import FFmpegNotFoundError
from ui.logger import loggerFactory


FrameProcessor = Callable[[str], np.ndarray]
WandProcessor = Callable[[str, Tuple[int, int], Dict[str, Any]], np.ndarray]


@dataclass
class ExportRuntimeOptions:
    image_format: str
    anim_format: str
    quality: int
    max_fps: int
    loop: bool
    target_path: Optional[str]
    direct_copy: bool
    profile: str
    diagnostic_id: Optional[str] = None


def clamp_fps(source_fps: int, max_fps: int) -> int:
    if source_fps <= 0:
        source_fps = 1
    if max_fps <= 0:
        return max(1, source_fps)
    return max(1, min(int(source_fps), int(max_fps)))


@contextmanager
def tempdir(prefix: str = "rmbg_"):
    path = tempfile.mkdtemp(prefix=prefix)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class InputRouterService:
    def __init__(
        self,
        *,
        remove_rembg: FrameProcessor,
        remove_openvino: FrameProcessor,
        remove_hsv: FrameProcessor,
        remove_wand: WandProcessor,
        passthrough: FrameProcessor,
    ) -> None:
        self._remove_rembg = remove_rembg
        self._remove_openvino = remove_openvino
        self._remove_hsv = remove_hsv
        self._remove_wand = remove_wand
        self._passthrough = passthrough

    def process(
        self,
        engine: str,
        path: str,
        *,
        wand_seed: Optional[Tuple[int, int]],
        wand_opts: Dict[str, Any],
    ) -> np.ndarray:
        engine = (engine or "rembg").lower()
        if engine == "openvino":
            return self._remove_openvino(path)
        if engine == "hsv":
            return self._remove_hsv(path)
        if engine == "wand":
            if not wand_seed:
                raise RuntimeError("魔術棒需要先在圖片上取樣（點擊）座標。")
            return self._remove_wand(path, wand_seed, wand_opts or {})
        if engine in ("none", "copy"):
            return self._passthrough(path)
        return self._remove_rembg(path)


class EncoderService:
    def __init__(self, logger) -> None:
        self.logger = logger

    def _log(
        self,
        level: int,
        event: str,
        options: Optional[ExportRuntimeOptions] = None,
        *,
        diagnostic_id: Optional[str] = None,
        **fields: object,
    ) -> None:
        diag = diagnostic_id or (options.diagnostic_id if options else None)
        log_structured(self.logger, level, diag, event, **fields)

    def ensure_ffmpeg(self, *, diagnostic_id: Optional[str] = None) -> None:
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            self._log(logging.DEBUG, "ffmpeg.ensure.ok", diagnostic_id=diagnostic_id)
        except Exception as exc:
            self._log(
                logging.ERROR,
                "ffmpeg.ensure.error",
                diagnostic_id=diagnostic_id,
                error=repr(exc),
            )
            raise attach_diagnostic(
                RuntimeError("需要 ffmpeg。請安裝並加入 PATH。"),
                diagnostic_id,
            ) from exc

    def copy(
        self, src: str, dest: str, options: Optional[ExportRuntimeOptions] = None
    ) -> str:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(src, dest)
        self._log(logging.INFO, "encoder.copy", options, src=src, dest=dest)
        return dest

    def resolve_image_target(
        self, src: str, out_dir: str, options: ExportRuntimeOptions
    ) -> str:
        if options.target_path:
            return options.target_path
        base = os.path.splitext(os.path.basename(src))[0]
        suffix = "_rmbg"
        ext = options.image_format
        if not ext.startswith("."):
            ext = f".{ext}"
        os.makedirs(out_dir, exist_ok=True)
        target = os.path.join(out_dir, f"{base}{suffix}{ext}")
        self._log(logging.DEBUG, "encoder.resolve.image", options, target=target)
        return target

    def resolve_anim_target(
        self, src: str, out_dir: str, options: ExportRuntimeOptions
    ) -> str:
        if options.target_path:
            return options.target_path
        base = os.path.splitext(os.path.basename(src))[0]
        suffix = "_rmbg"
        ext = options.anim_format
        if not ext.startswith("."):
            ext = f".{ext}"
        os.makedirs(out_dir, exist_ok=True)
        base = base.split("-")[0]
        target = os.path.join(out_dir, f"{base}{suffix}{ext}")
        self._log(logging.DEBUG, "encoder.resolve.anim", options, target=target)
        return target

    def encode_image(
        self, rgba: np.ndarray, target_path: str, options: ExportRuntimeOptions
    ) -> str:
        Image.fromarray(rgba, "RGBA").save(target_path)
        self._log(logging.INFO, "encoder.encode.image", options, target=target_path)
        return target_path

    def encode_webp(
        self, rgba: np.ndarray, target_path: str, options: ExportRuntimeOptions
    ) -> str:
        self.ensure_ffmpeg(diagnostic_id=options.diagnostic_id)
        with tempdir(prefix="rmbg_encode_") as tmp:
            tmp_png = os.path.join(tmp, "frame.png")
            Image.fromarray(rgba, "RGBA").save(tmp_png)
            args = [
                "ffmpeg",
                "-y",
                "-i",
                tmp_png,
                "-c:v",
                "libwebp",
                "-lossless",
                "1",
                "-compression_level",
                "6",
                "-preset",
                "picture",
                target_path,
            ]
            self._log(
                logging.INFO,
                "ffmpeg.encode.webp.start",
                options,
                command=args,
                target=target_path,
            )
            try:
                subprocess.run(
                    args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
                )
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or b"").decode("utf-8", "ignore")
                self._log(
                    logging.ERROR,
                    "ffmpeg.encode.webp.error",
                    options,
                    returncode=exc.returncode,
                    stderr=stderr,
                )
                raise attach_diagnostic(
                    RuntimeError("ffmpeg encode webp 失敗"), options.diagnostic_id
                ) from exc
        self._log(
            logging.INFO, "ffmpeg.encode.webp.success", options, target=target_path
        )
        return target_path

    def probe_fps(
        self, path: str, options: Optional[ExportRuntimeOptions] = None
    ) -> int:
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=avg_frame_rate",
                    "-of",
                    "default=nw=1:nk=1",
                    path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
            num, den = result.stdout.strip().split("/")
            num = int(num)
            den = int(den) if int(den) != 0 else 1
            fps = max(1, round(num / den))
            self._log(logging.DEBUG, "ffprobe.fps", options, path=path, fps=fps)
            return fps
        except Exception as exc:
            self._log(
                logging.WARNING,
                "ffprobe.fps.fallback",
                options,
                path=path,
                error=repr(exc),
            )
            return 15

    def extract_frames(
        self,
        path: str,
        frames_dir: str,
        range_ms: Tuple[Optional[int], Optional[int]],
        options: ExportRuntimeOptions,
    ) -> None:
        self.ensure_ffmpeg(diagnostic_id=options.diagnostic_id)
        os.makedirs(frames_dir, exist_ok=True)
        t_in, t_out = range_ms
        args = ["ffmpeg", "-y"]
        if isinstance(t_in, int) and t_in > 0:
            args += ["-ss", f"{t_in / 1000:.3f}"]
        args += ["-i", path, "-vsync", "0"]
        if isinstance(t_out, int) and t_out > 0:
            args += ["-to", f"{t_out / 1000:.3f}"]
        args += [os.path.join(frames_dir, "f_%06d.png")]
        self._log(
            logging.INFO,
            "ffmpeg.extract.start",
            options,
            command=args,
            frames_dir=frames_dir,
        )
        try:
            subprocess.run(
                args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"").decode("utf-8", "ignore")
            self._log(
                logging.ERROR,
                "ffmpeg.extract.error",
                options,
                returncode=exc.returncode,
                stderr=stderr,
            )
            raise attach_diagnostic(
                RuntimeError("ffmpeg 抽幀失敗"), options.diagnostic_id
            ) from exc

    def assemble_animation(
        self,
        frames_dir: str,
        fps: int,
        options: ExportRuntimeOptions,
        target_path: str,
    ) -> str:
        self.ensure_ffmpeg(diagnostic_id=options.diagnostic_id)
        loop_flag = "0" if options.loop else "1"
        quality = str(options.quality)
        args = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(max(1, fps)),
            "-i",
            os.path.join(frames_dir, "f_%06d.png"),
            "-c:v",
            "libwebp_anim",
            "-pix_fmt",
            "yuva420p",
            "-loop",
            loop_flag,
            "-q:v",
            quality,
            target_path,
        ]
        self._log(
            logging.INFO,
            "ffmpeg.assemble.start",
            options,
            command=args,
            target=target_path,
        )
        try:
            subprocess.run(
                args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"").decode("utf-8", "ignore")
            self._log(
                logging.ERROR,
                "ffmpeg.assemble.error",
                options,
                returncode=exc.returncode,
                stderr=stderr,
            )
            raise attach_diagnostic(
                RuntimeError("ffmpeg 合成失敗"), options.diagnostic_id
            ) from exc
        self._log(
            logging.INFO,
            "ffmpeg.assemble.success",
            options,
            frames_dir=frames_dir,
            target=target_path,
        )
        return target_path


class ImagePipeline:
    def __init__(
        self, router: InputRouterService, encoder: EncoderService, logger
    ) -> None:
        self.router = router
        self.encoder = encoder
        self.logger = loggerFactory(
            logger_name=self.__class__.__name__, log_level=logger.level
        ).getLogger()

    def _log(
        self, level: int, event: str, options: ExportRuntimeOptions, **fields: object
    ) -> None:
        log_structured(self.logger, level, options.diagnostic_id, event, **fields)

    def run(
        self,
        src: str,
        out_dir: str,
        engine: str,
        options: ExportRuntimeOptions,
        *,
        wand_seed: Optional[Tuple[int, int]],
        wand_opts: Dict[str, Any],
        progress_cb: Callable[[int, str], None],
    ) -> str:
        self._log(logging.INFO, "pipeline.image.start", options, src=src, engine=engine)
        if options.direct_copy and options.target_path:
            progress_cb(100, "done(copy)")
            self._log(
                logging.INFO, "pipeline.image.copy", options, target=options.target_path
            )
            return self.encoder.copy(src, options.target_path, options)

        try:
            rgba = self.router.process(
                engine, src, wand_seed=wand_seed, wand_opts=wand_opts or {}
            )
        except Exception as exc:
            self._log(
                logging.ERROR, "pipeline.image.router.error", options, error=repr(exc)
            )
            raise attach_diagnostic(exc, options.diagnostic_id)

        target = self.encoder.resolve_image_target(src, out_dir, options)
        if target.lower().endswith(".webp"):
            result = self.encoder.encode_webp(rgba, target, options)
        else:
            result = self.encoder.encode_image(rgba, target, options)
        progress_cb(100, "done")
        self._log(logging.INFO, "pipeline.image.done", options, target=result)
        return result


class VideoPipeline:
    def __init__(
        self, router: InputRouterService, encoder: EncoderService, logger
    ) -> None:
        self.router = router
        self.encoder = encoder
        self.logger = loggerFactory(
            logger_name=self.__class__.__name__, log_level=logger.level
        ).getLogger()

    def _log(
        self, level: int, event: str, options: ExportRuntimeOptions, **fields: object
    ) -> None:
        log_structured(self.logger, level, options.diagnostic_id, event, **fields)

    def run(
        self,
        src: str,
        out_dir: str,
        engine: str,
        options: ExportRuntimeOptions,
        *,
        wand_seed: Optional[Tuple[int, int]],
        wand_opts: Dict[str, Any],
        range_ms: Tuple[Optional[int], Optional[int]],
        progress_cb: Callable[[int, str], None],
    ) -> Dict[str, Any]:
        self._log(
            logging.INFO,
            "pipeline.video.start",
            options,
            src=src,
            engine=engine,
            range=range_ms,
        )
        if (
            options.direct_copy
            and options.target_path
            and all(ms in (None, 0) for ms in range_ms)
        ):
            progress_cb(100, "done(copy)")
            target = self.encoder.copy(src, options.target_path, options)
            self._log(logging.INFO, "pipeline.video.copy", options, target=target)
            return {
                "input": src,
                "output": target,
                "kind": "anim",
                "frames": None,
                "fps": None,
            }

        target_path = self.encoder.resolve_anim_target(src, out_dir, options)
        with tempdir(prefix="rmbg_") as tmp:
            frames_dir = os.path.join(tmp, "frames")
            out_frames = os.path.join(tmp, "out")
            os.makedirs(out_frames, exist_ok=True)

            try:
                self.encoder.extract_frames(src, frames_dir, range_ms, options)
            except Exception as exc:
                self._log(
                    logging.ERROR,
                    "pipeline.video.extract.error",
                    options,
                    error=repr(exc),
                )
                raise
            self._log(
                logging.INFO,
                "pipeline.video.extract.done",
                options,
                frames_dir=frames_dir,
            )
            files = sorted(
                f for f in os.listdir(frames_dir) if f.lower().endswith(".png")
            )
            total = max(1, len(files))

            for idx, fname in enumerate(files, 1):
                if idx % 3 == 0 or idx == total:
                    pct = 5 + int(80 * idx / total)
                    progress_cb(pct, f"frame {idx}/{total}")
                frame_path = os.path.join(frames_dir, fname)
                try:
                    rgba = self.router.process(
                        engine,
                        frame_path,
                        wand_seed=wand_seed,
                        wand_opts=wand_opts or {},
                    )
                except Exception as exc:
                    self._log(
                        logging.ERROR,
                        "pipeline.video.router.error",
                        options,
                        frame=fname,
                        error=repr(exc),
                    )
                    raise attach_diagnostic(exc, options.diagnostic_id)
                Image.fromarray(rgba, "RGBA").save(os.path.join(out_frames, fname))

            fps = self.encoder.probe_fps(src, options)
            fps = clamp_fps(fps, options.max_fps)
            progress_cb(90, "encode")
            try:
                self.encoder.assemble_animation(out_frames, fps, options, target_path)
            except Exception:
                self._log(
                    logging.ERROR,
                    "pipeline.video.assemble.error",
                    options,
                    target=target_path,
                )
                raise

        progress_cb(100, "done")
        self._log(
            logging.INFO,
            "pipeline.video.done",
            options,
            target=target_path,
            frames=len(files),
            fps=fps,
        )
        return {
            "input": src,
            "output": target_path,
            "kind": "anim",
            "frames": len(files),
            "fps": fps,
        }
