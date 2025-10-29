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
