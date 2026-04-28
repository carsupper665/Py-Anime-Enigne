from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from typing import Callable, Optional

from core.diagnostics import log_structured
from core.export_context import ExportOptions
from core.job_models import JobRequest


@dataclass
class SaveDialogResult:
    filename: str
    options: ExportOptions


@dataclass
class SavePlan:
    queue_job: JobRequest


class SaveExportController:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        config_provider: Callable[[], dict],
        config_saver: Callable[[dict], None],
        enqueue_job: Callable[[JobRequest], None],
        toast_emitter: Callable[[dict], None],
        get_out_dir: Callable[[], str],
        confirm_overwrite: Callable[[str], bool],
        show_message: Callable[[str, str], None],
        can_direct_copy: Callable[[str, str], bool],
        build_queue_job: Callable[
            [str, str, ExportOptions, Optional[str], bool], Optional[JobRequest]
        ],
        on_options_updated: Optional[Callable[[ExportOptions], None]] = None,
    ) -> None:
        self._logger = logger
        self._config_provider = config_provider
        self._config_saver = config_saver
        self._enqueue_job = enqueue_job
        self._toast_emitter = toast_emitter
        self._get_out_dir = get_out_dir
        self._confirm_overwrite = confirm_overwrite
        self._show_message = show_message
        self._can_direct_copy = can_direct_copy
        self._build_queue_job = build_queue_job
        self._on_options_updated = on_options_updated

    def persist_output_settings(self, options: ExportOptions) -> None:
        cfg = self._config_provider()
        cfg.setdefault("export", {})
        cfg["export"].setdefault("options", {})
        cfg["export"]["options"].update(options.to_dict())
        cfg.setdefault("output", {})
        cfg["output"]["quality"] = options.quality
        cfg["output"]["max_fps"] = options.max_fps
        cfg["output"]["loop"] = options.loop
        try:
            self._config_saver(cfg)
        except Exception:
            pass

    def build_save_plan(
        self, current_path: Optional[str], result: SaveDialogResult, rem_bg: bool
    ) -> Optional[SavePlan]:
        if not current_path:
            return None
        out_dir = self._get_out_dir()
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, result.filename)

        if os.path.abspath(save_path) == os.path.abspath(current_path):
            self._show_message("提示", "來源與目標相同。")
            return None
        if not self._confirm_overwrite(save_path):
            return None

        if rem_bg:
            queue_payload = self._build_queue_job(
                current_path, save_path, result.options, None, True
            )
            if queue_payload:
                return SavePlan(queue_job=queue_payload)
            return None

        direct_copy = self._can_direct_copy(current_path, save_path)
        adjusted_options = replace(
            result.options,
            target_path=save_path,
            direct_copy=direct_copy,
            profile=result.options.profile,
        )
        if self._on_options_updated:
            self._on_options_updated(adjusted_options)
        queue_payload = self._build_queue_job(
            current_path, save_path, adjusted_options, "none", False
        )
        if queue_payload:
            return SavePlan(queue_job=queue_payload)
        return None

    def run_save_plan(self, plan: SavePlan) -> None:
        payload = plan.queue_job
        self._enqueue_job(payload)
        log_structured(
            self._logger,
            logging.INFO,
            payload.diagnostic_id,
            "queue.job.enqueued",
            src=payload.src,
            prefer=payload.prefer,
        )
        self._toast_emitter(
            {
                "level": "info",
                "title": "已加入佇列",
                "message": f"等待處理…（追蹤 {payload.diagnostic_id}）",
                "duration": 3000,
            }
        )
