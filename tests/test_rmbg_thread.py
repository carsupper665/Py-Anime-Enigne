import os
import logging

import pytest

pytest.importorskip("pytestqt")
from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot


@pytest.mark.qt
def test_rmbg_thread_finished_emits_once(qtbot, monkeypatch, tmp_path):
    from ui import threads

    class DummyWorker(QObject):
        progress = pyqtSignal(dict)
        finished = pyqtSignal(dict)
        error = pyqtSignal(object)

        def __init__(self, *args, **kwargs):
            super().__init__()
            self._canceled = False

        @pyqtSlot()
        def do_remove(self):
            QTimer.singleShot(0, lambda: self.finished.emit({"ok": True}))

        def request_cancel(self):
            self._canceled = True

    monkeypatch.setattr(threads, "_RmbgWorker", DummyWorker)
    monkeypatch.setattr(os.path, "isfile", lambda _: True)
    monkeypatch.setattr(os, "makedirs", lambda *args, **kwargs: None)

    rmbg = threads.RmbgThread(logging.getLogger("test"))
    results = []
    rmbg.finished.connect(results.append)

    rmbg.remove_bg(str(tmp_path / "input.png"), str(tmp_path / "out"), "hsv", "hsv")
    qtbot.waitUntil(lambda: len(results) == 1, timeout=1000)
    assert len(results) == 1


@pytest.mark.qt
def test_rmbg_thread_handles_multiple_jobs(qtbot, monkeypatch, tmp_path):
    from ui import threads

    class DummyWorker(QObject):
        progress = pyqtSignal(dict)
        finished = pyqtSignal(dict)
        error = pyqtSignal(object)

        def __init__(self, *args, **kwargs):
            super().__init__()
            self._diag = kwargs.get("diagnostic_id")

        @pyqtSlot()
        def do_remove(self):
            payload = {"ok": True, "diagnostic_id": self._diag}
            QTimer.singleShot(0, lambda: self.finished.emit(payload))

        def request_cancel(self):
            pass

    monkeypatch.setattr(threads, "_RmbgWorker", DummyWorker)
    monkeypatch.setattr(os.path, "isfile", lambda _: True)
    monkeypatch.setattr(os, "makedirs", lambda *args, **kwargs: None)

    rmbg = threads.RmbgThread(logging.getLogger("test"))
    finishes: list[str] = []
    rmbg.finished.connect(lambda payload: finishes.append(payload.get("diagnostic_id")))

    diag1 = rmbg.remove_bg(
        str(tmp_path / "a.png"), str(tmp_path / "out1"), "hsv", "hsv"
    )
    diag2 = rmbg.remove_bg(
        str(tmp_path / "b.png"), str(tmp_path / "out2"), "hsv", "hsv"
    )

    assert diag1 != diag2
    qtbot.waitUntil(lambda: len(finishes) == 2, timeout=1000)
    assert sorted(finishes) == sorted([diag1, diag2])

    # 所有工作應在完成後清理
    assert not rmbg._jobs

    # cancel_current 不應出錯（即使無工作）
    rmbg.cancel_current()
