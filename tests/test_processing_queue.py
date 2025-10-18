import time
import types

from ui.queue import ProcessingQueue, QueueJob


def test_queue_basic_order(qtbot):
    q = ProcessingQueue()

    events = []

    def on_started(job):
        events.append(("start", job.src))
        # 模擬立即完成
        q.notify_finished({"ok": True})

    q.job_started.connect(on_started)

    q.enqueue(QueueJob(src="a.png", prefer="hsv", opts={}))
    q.enqueue(QueueJob(src="b.png", prefer="hsv", opts={}))

    # 讓事件迴圈處理一下
    qtbot.waitUntil(lambda: events == [("start", "a.png"), ("start", "b.png")], timeout=1000)

    assert events == [("start", "a.png"), ("start", "b.png")]


def test_queue_pause_resume(qtbot):
    q = ProcessingQueue()
    events = []
    q.pause()

    def on_started(job):
        events.append(("start", job.src))
        q.notify_finished({"ok": True})

    q.job_started.connect(on_started)

    q.enqueue(QueueJob(src="x.png", prefer="hsv", opts={}))
    # 暫停時不應該啟動
    with qtbot.assertNotEmitted(q.job_started, timeout=200):
        pass
    assert events == []

    q.resume()
    qtbot.waitUntil(lambda: events == [("start", "x.png")], timeout=1000)

