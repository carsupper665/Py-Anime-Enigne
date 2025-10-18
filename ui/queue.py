from __future__ import annotations
"""
OpenSpec: add-processing-queue
spec: openspec/changes/add-processing-queue/specs/processing-queue/spec.md:3
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from PyQt6.QtCore import QObject, pyqtSignal


@dataclass
class QueueJob:
    src: str
    prefer: str
    opts: Dict[str, Any]


class ProcessingQueue(QObject):
    job_enqueued = pyqtSignal(object)      # QueueJob
    job_started = pyqtSignal(object)       # QueueJob
    job_progress = pyqtSignal(object, dict)  # (QueueJob, progress)
    job_finished = pyqtSignal(object, dict)  # (QueueJob, payload)
    job_error = pyqtSignal(object, object)   # (QueueJob, exc)
    queue_empty = pyqtSignal()
    state_changed = pyqtSignal(str)  # idle|running|paused

    def __init__(self):
        super().__init__()
        self._q: List[QueueJob] = []
        self._current: Optional[QueueJob] = None
        self._paused: bool = False

    # public API
    def enqueue(self, job: QueueJob):
        self._q.append(job)
        self.job_enqueued.emit(job)
        if not self._current and not self._paused:
            self._start_next()

    def cancel_current(self):
        # 僅阻止下一個任務；現有任務交由外部工作者響應取消（如有）
        self._current = None

    def clear_pending(self):
        self._q.clear()
        if not self._current:
            self.queue_empty.emit()

    def pause(self):
        if not self._paused:
            self._paused = True
            self.state_changed.emit("paused")

    def resume(self):
        if self._paused:
            self._paused = False
            self.state_changed.emit("running" if self._current else "idle")
            if not self._current:
                self._start_next()

    # to be called by owner when worker updates
    def notify_progress(self, p: dict):
        if self._current:
            self.job_progress.emit(self._current, p)

    def notify_finished(self, payload: dict):
        if self._current:
            self.job_finished.emit(self._current, payload)
        self._current = None
        if self._paused:
            self.state_changed.emit("paused")
            return
        self._start_next()

    def notify_error(self, exc: object):
        if self._current:
            self.job_error.emit(self._current, exc)
        self._current = None
        if self._paused:
            self.state_changed.emit("paused")
            return
        self._start_next()

    # internal
    def _start_next(self):
        if self._paused:
            self.state_changed.emit("paused")
            return
        if self._current is not None:
            return
        if not self._q:
            self.state_changed.emit("idle")
            self.queue_empty.emit()
            return
        self._current = self._q.pop(0)
        self.state_changed.emit("running")
        self.job_started.emit(self._current)
