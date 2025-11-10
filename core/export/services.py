from __future__ import annotations

import logging
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from core.export_context import ExportOptions
from core.diagnostics import log_structured


class TempDirectoryManager:
    """Manage unique temporary files under a shared directory."""

    def __init__(
        self, base_dir: str = "./temp", logger: Optional[logging.Logger] = None
    ) -> None:
        self.base_path = Path(base_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._tracked: set[Path] = set()
        self._logger = logger or logging.getLogger(__name__)

    def _unique_name(self, original_name: str) -> str:
        base = Path(original_name)
        suffix = base.suffix or ".tmp"
        stem = base.stem or "temp"
        return f"{stem}-{uuid.uuid4().hex}{suffix}"

    def _log(self, event: str, diagnostic_id: Optional[str], **fields: object) -> None:
        log_structured(self._logger, logging.INFO, diagnostic_id, event, **fields)

    def copy_to_temp(
        self,
        src: str,
        preferred_name: Optional[str] = None,
        persistent: bool = True,
        *,
        diagnostic_id: Optional[str] = None,
    ) -> str:
        name = preferred_name or Path(src).name
        target = self.base_path / self._unique_name(name)
        shutil.copyfile(src, target)
        if not persistent:
            self._tracked.add(target)
        self._log(
            "temp.copy",
            diagnostic_id,
            src=src,
            target=str(target),
            persistent=persistent,
        )
        return str(target)

    def register(self, path: str, *, diagnostic_id: Optional[str] = None) -> None:
        p = Path(path)
        self._tracked.add(p)
        self._log("temp.register", diagnostic_id, path=str(p))

    def cleanup(
        self,
        paths: Optional[Iterable[str]] = None,
        *,
        diagnostic_id: Optional[str] = None,
    ) -> None:
        candidates = paths if paths is not None else [str(p) for p in self._tracked]
        for item in candidates:
            try:
                p = Path(item)
                if p.exists():
                    p.unlink()
                    self._log("temp.cleanup.success", diagnostic_id, path=str(p))
            except Exception as exc:
                self._log(
                    "temp.cleanup.error", diagnostic_id, path=item, error=repr(exc)
                )
                continue
            finally:
                self._tracked.discard(Path(item))


class ExportCommandBuilder:
    """Build and customise ffmpeg command arguments."""

    PROFILE_WEBP = "webp"

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def _log(self, event: str, diagnostic_id: Optional[str], **fields: object) -> None:
        log_structured(self._logger, logging.INFO, diagnostic_id, event, **fields)

    def apply_profile(
        self,
        args: List[str],
        profile: str,
        options: ExportOptions,
        *,
        diagnostic_id: Optional[str] = None,
    ) -> List[str]:
        if profile == self.PROFILE_WEBP:
            profiled = self._apply_webp_profile(
                args, options, diagnostic_id=diagnostic_id
            )
        else:
            profiled = list(args)
        self._log(
            "command.profile",
            diagnostic_id,
            profile=profile,
            options=options.to_dict(),
            before=args,
            after=profiled,
        )
        return profiled

    def _apply_webp_profile(
        self,
        args: List[str],
        options: ExportOptions,
        *,
        diagnostic_id: Optional[str] = None,
    ) -> List[str]:
        new_args = list(args)
        has_anim = any(token == "libwebp_anim" for token in new_args)
        if not has_anim:
            return new_args

        # loop flag
        loop_flag = "0" if options.loop else "1"
        if "-loop" in new_args:
            idx = new_args.index("-loop")
            if idx + 1 < len(new_args):
                new_args[idx + 1] = loop_flag
        else:
            new_args.extend(["-loop", loop_flag])

        # quality
        if "-q:v" in new_args:
            idx = new_args.index("-q:v")
            if idx + 1 < len(new_args):
                new_args[idx + 1] = str(options.quality)
        else:
            new_args.extend(["-q:v", str(options.quality)])

        # fps limiter
        has_filter = "-filter:v" in new_args or any(
            isinstance(token, str) and token.startswith("fps=") for token in new_args
        )
        if options.max_fps > 0 and not has_filter:
            new_args.extend(["-filter:v", f"fps={options.max_fps}"])

        return new_args

    def build_trim_command(
        self,
        src: str,
        start_ms: int,
        end_ms: int,
        dest: str,
        *,
        diagnostic_id: Optional[str] = None,
    ) -> List[str]:
        args = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_ms / 1000:.3f}",
            "-to",
            f"{end_ms / 1000:.3f}",
            "-i",
            src,
            "-c",
            "copy",
            dest,
        ]
        self._log(
            "command.build.trim",
            diagnostic_id,
            src=src,
            dest=dest,
            start_ms=start_ms,
            end_ms=end_ms,
            command=args,
        )
        return args

    def build_mute_command(
        self,
        src: str,
        dest: str,
        *,
        diagnostic_id: Optional[str] = None,
    ) -> List[str]:
        args = ["ffmpeg", "-y", "-i", src, "-c", "copy", "-an", dest]
        self._log("command.build.mute", diagnostic_id, src=src, dest=dest, command=args)
        return args


Runner = Callable[[List[str]], Tuple[bool, str]]


def default_runner(args: List[str]) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        return True, proc.stderr.decode("utf-8", "ignore")
    except subprocess.CalledProcessError as exc:
        return False, exc.stderr.decode("utf-8", "ignore")


@dataclass
class CommandResult:
    success: bool
    log: str


class VideoExportService:
    def __init__(
        self,
        runner: Runner = default_runner,
        command_builder: Optional[ExportCommandBuilder] = None,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._runner = runner
        self._logger = logger or logging.getLogger(__name__)
        self._builder = command_builder or ExportCommandBuilder(logger=self._logger)

    def _log(self, event: str, diagnostic_id: Optional[str], **fields: object) -> None:
        log_structured(self._logger, logging.INFO, diagnostic_id, event, **fields)

    def trim(
        self,
        src: str,
        start_ms: int,
        end_ms: int,
        dest: str,
        options: ExportOptions,
        *,
        diagnostic_id: Optional[str] = None,
    ) -> CommandResult:
        args = self._builder.build_trim_command(
            src, start_ms, end_ms, dest, diagnostic_id=diagnostic_id
        )
        args = self._builder.apply_profile(
            args,
            ExportCommandBuilder.PROFILE_WEBP,
            options,
            diagnostic_id=diagnostic_id,
        )
        ok, log = self._runner(args)
        self._log("command.run.trim", diagnostic_id, success=ok, command=args)
        return CommandResult(success=ok, log=log)

    def mute(
        self,
        src: str,
        dest: str,
        options: ExportOptions,
        *,
        diagnostic_id: Optional[str] = None,
    ) -> CommandResult:
        args = self._builder.build_mute_command(src, dest, diagnostic_id=diagnostic_id)
        args = self._builder.apply_profile(
            args,
            ExportCommandBuilder.PROFILE_WEBP,
            options,
            diagnostic_id=diagnostic_id,
        )
        ok, log = self._runner(args)
        self._log("command.run.mute", diagnostic_id, success=ok, command=args)
        return CommandResult(success=ok, log=log)

    def run_custom(
        self,
        args: List[str],
        options: Optional[ExportOptions] = None,
        *,
        diagnostic_id: Optional[str] = None,
    ) -> CommandResult:
        final_args = list(args)
        if options is not None:
            final_args = self._builder.apply_profile(
                final_args,
                ExportCommandBuilder.PROFILE_WEBP,
                options,
                diagnostic_id=diagnostic_id,
            )
        ok, log = self._runner(final_args)
        self._log("command.run.custom", diagnostic_id, success=ok, command=final_args)
        return CommandResult(success=ok, log=log)
