from __future__ import annotations

import logging
import os
from typing import Callable, Optional

from core.diagnostics import log_structured
from ui.queue import QueueJob


class QueueCoordinator:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        toast_loading,
        home_page,
        run_job: Callable[[QueueJob], None],
    ) -> None:
        self._logger = logger
        self._toast_loading = toast_loading
        self._home_page = home_page
        self._run_job = run_job
        self._queue_pending = 0
        self._queue_started_count = 0
        self._suppress_next_loading = False

    def consume_loading_suppression(self) -> bool:
        if self._suppress_next_loading:
            self._suppress_next_loading = False
            return True
        return False

    def on_job_enqueued(self, _job: QueueJob) -> None:
        self._update_queue_ui(delta=1)

    def on_job_finished(self, _job: QueueJob, _payload: dict) -> None:
        self._update_queue_ui(delta=-1)

    def on_job_error(self, _job: QueueJob, _exc: object) -> None:
        self._update_queue_ui(delta=-1)

    def on_queue_empty(self) -> None:
        try:
            self._queue_started_count = 0
            self._toast_loading.on_exception_cancel()
        except Exception:
            pass
        self._set_queue_label(0)

    def on_job_started(self, job: QueueJob) -> None:
        try:
            self._queue_started_count = getattr(self, "_queue_started_count", 0) + 1
            total = max(1, getattr(self, "_queue_pending", 0) + 1)
            title = f"Processing {self._queue_started_count}/{total}"
            msg = os.path.basename(getattr(job, "src", ""))
            diagnostic_id = getattr(job, "diagnostic_id", None)
            if not diagnostic_id and isinstance(getattr(job, "opts", None), dict):
                diagnostic_id = job.opts.get("diagnostic_id")
            if diagnostic_id:
                msg = f"{msg} (ID {diagnostic_id})"
            log_structured(
                self._logger,
                logging.INFO,
                diagnostic_id,
                "queue.job.started",
                src=getattr(job, "src", ""),
                prefer=getattr(job, "prefer", ""),
            )
            self._suppress_next_loading = True
            self._toast_loading.show_loading(title=title, message=msg)
            self._run_job(job)
        except Exception:
            pass

    def _set_queue_label(self, n: int) -> None:
        self._queue_pending = max(0, n)
        try:
            if hasattr(self._home_page, "queue_label"):
                self._home_page.queue_label.setText(
                    f"Queue: {self._queue_pending} pending"
                )
        except Exception:
            pass

    def _update_queue_ui(self, delta: int = 0) -> None:
        self._set_queue_label(self._queue_pending + delta)
