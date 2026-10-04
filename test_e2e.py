"""End to end, real app + real camera thread on generated clips: python test_e2e.py"""
import contextlib
import os
import pathlib
import resource
import shutil
import sys
import time
import types
import urllib.error
from unittest import mock

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import capture
import rehapose
import storage
from PyQt5 import QtCore, QtTest, QtWidgets, sip
from test_rehapose import RECORD, sample_frame, sandbox, save_artifacts, upper_body_frame

FPS = 30.0
CLIPS = {}  # name -> frames that FakeCapture serves in place of a camera
REAL_CAPTURE = cv2.VideoCapture
HOLD = rehapose.SETUP_HOLD


class FakeCapture:
    """Serves CLIPS by name exactly like a video file; refuses any camera index."""

    opened = []

    def __init__(self, source, *_a):
        FakeCapture.opened.append(source)
        assert isinstance(source, str), f"a test opened camera {source}"
        self.frames = list(CLIPS.get(source, ()))
        self.ok = source in CLIPS

    def isOpened(self):
        return self.ok

    def set(self, *_a):
        return True

    def read(self):
        return (True, self.frames.pop(0).copy()) if self.frames else (False, None)

    def release(self):
        self.frames = []


def guarded_capture(source, *a):
    assert isinstance(source, str), f"a test opened camera {source}"
    return REAL_CAPTURE(source, *a)


class SlowDownload:
    """A model download that stalls for 3 s and then delivers nothing."""

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def read(self, *_a):
        time.sleep(3)
        return b""


def pump(until, seconds):
    deadline = time.monotonic() + seconds
    while not until() and time.monotonic() < deadline:
        QtTest.QTest.qWait(20)
    return until()


def trace_rows(path):
    return path.read_text(encoding="utf-8-sig").split("\ntime_s,", 1)[1].count("\n") - 1


@contextlib.contextmanager
def app_on(clip, exercise, tmp, name, gui_delay=0.0, **patches):
    """A fresh window on `clip`: lite pinned, video-time clock, sessions in their own folder."""
    clock, counts = [0.0], {"live": 0, "stale": 0}
    real_on_frame = rehapose.Main.on_frame

    def on_frame(self, *args):  # the clock moves one frame per delivered frame: exact results
        clock[0] += 1 / FPS
        counts["live" if self.worker is not None else "stale"] += 1
        if gui_delay:
            time.sleep(gui_delay)  # a slow GUI, so frames queue up behind it
        return real_on_frame(self, *args)

    storage.settings().clear()
    storage.settings().setValue("dataDir", str(pathlib.Path(tmp) / name))
    RECORD.dialogs.clear()
    RECORD.tones.clear()
    defaults = {"TIERS": ("lite",)}
    with contextlib.ExitStack() as stack:
        for target, attr, value in (
                (capture.cv2, "VideoCapture", FakeCapture),
                (rehapose, "time", types.SimpleNamespace(perf_counter=lambda: clock[0])),
                (rehapose.Main, "on_frame", on_frame),
                *((capture, k, v) for k, v in {**defaults, **patches}.items() if k.isupper()),
                *((rehapose, k[9:], v) for k, v in patches.items() if k.startswith("rehapose_")),
                *(patches.get("extra", ()))):
            stack.enter_context(mock.patch.object(target, attr, value))
        window = rehapose.Main(camera=clip)
        statuses = []
        real_set = window.status.setText
        window.status.setText = lambda text: (statuses.append(text), real_set(text))[1]
        window.show()
        window.exercise.setCurrentIndex(window.exercise.findData(exercise))
        run = types.SimpleNamespace(window=window, statuses=statuses, counts=counts,
                                    folder=pathlib.Path(tmp) / name / "sessions")
        try:
            yield run
        except BaseException:
            save_artifacts(f"e2e-{name}", tmp)  # before shut_down closes the window
            raise
        finally:
            shut_down(window)


def shut_down(window):
    """Never let a running QThread be destroyed: that is a qFatal abort, not a test failure."""
    for worker in (window.worker, window.retiring):
        if worker is not None and not sip.isdeleted(worker):
            worker.stop()
            if not worker.wait(10000):
                print("a worker would not stop; exiting before Qt can abort", file=sys.stderr)
                os._exit(1)
    window.worker = None
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)  # qWait won't
    QtTest.QTest.qWait(50)


def press_start_and_finish(run, seconds=60):
    QtTest.QTest.mouseClick(run.window.button, QtCore.Qt.LeftButton)
    assert pump(lambda: run.window.worker is None, seconds), "the session never ended"
    QtTest.QTest.qWait(500)  # anything still queued gets its chance to misbehave
    files = sorted(run.folder.glob("*.csv"))
    saved = trace_rows(files[-1]) if files else 0
    assert len(run.window.times["left_knee"]) == saved, "frames recorded after the stop"
    return files


def upper_body_records(tmp):
    """The device bug: seated, head to hips, must record for every non-scored exercise."""
    for exercise in ("shoulder_abduction", "knee_flexion", "other"):
        with app_on("upper", exercise, tmp, f"upper-{exercise}") as run:
            files = press_start_and_finish(run)
            assert len(files) == 1, (exercise, run.statuses[-3:])
            head = storage.read_header(files[0])
            assert head["exercise"] == exercise and head["model"] == "lite", head
            assert trace_rows(files[0]) == len(CLIPS["upper"]) - HOLD + 1, trace_rows(files[0])
            assert "Saved" in run.statuses[-1], run.statuses[-1]
            assert not RECORD.dialogs, RECORD.dialogs
            if exercise == "knee_flexion":
                assert "Get your knees fully in frame." in " ".join(run.statuses), "no advice"
                check_history(run.window, files[0], exercise)


def check_history(window, path, exercise):
    window.show_history()
    table = window.sessions
    assert table.item(0, 2).text() == rehapose.EXERCISE_DISPLAY[exercise], table.item(0, 2).text()
    centre = table.visualItemRect(table.item(0, 0)).center()
    QtTest.QTest.mouseClick(table.viewport(), QtCore.Qt.LeftButton, pos=centre)
    QtTest.QTest.mouseDClick(table.viewport(), QtCore.Qt.LeftButton, pos=centre)
    assert window.viewing_stored == path, "double-click did not open the stored session"


def empty_room_saves_nothing(tmp):
    with app_on("empty", "knee_flexion", tmp, "empty") as run:
        assert press_start_and_finish(run) == []
        assert "you were never in view" in run.statuses[-1], run.statuses[-1]
        assert not RECORD.dialogs, RECORD.dialogs


def chair_stand_flow(tmp):
    """Countdown, 2 s test, automatic stop - with a slow GUI so frames back up behind it."""
    with app_on("full", "chair_stand_30s", tmp, "chair", gui_delay=0.01,
                rehapose_CHAIR_STAND_SECONDS=2.0) as run:
        files = press_start_and_finish(run)
        assert len(files) == 1, "the automatic stop wrote no file, or a second one"
        head = storage.read_header(files[0])
        assert head["complete"] == "yes" and "stands" in head, head
        assert trace_rows(files[0]) == 2 * FPS + 1, trace_rows(files[0])
        assert RECORD.tones == ["tick"] * 3 + ["go", "end"], RECORD.tones
        assert run.statuses[-1].startswith("Chair stands: 0."), run.statuses[-1]
        assert not RECORD.dialogs, RECORD.dialogs
        assert run.counts["live"] < len(CLIPS["full"]), "the test did not stop itself"
        print(f"      {run.counts['stale']} frames arrived after the stop, all dropped")


def quit_while_recording(tmp):
    box = QtWidgets.QMessageBox
    answer = staticmethod(lambda *a, **_k: RECORD.dialogs.append(("question", a[2])) or box.Save)
    with app_on("full", "knee_flexion", tmp, "quit", extra=[(box, "question", answer)]) as run:
        QtTest.QTest.mouseClick(run.window.button, QtCore.Qt.LeftButton)
        assert pump(lambda: len(run.window.times["left_knee"]) >= 30, 60), "never recorded"
        quit_action = next(a for m in run.window.menuBar().actions() if m.menu()
                           for a in m.menu().actions() if a.text() == "Quit")
        quit_action.trigger()
        assert run.window.worker is None and not run.window.isVisible(), "did not quit"
        assert len(list(run.folder.glob("*.csv"))) == 1, "quitting lost the session"
        assert [kind for kind, _ in RECORD.dialogs] == ["question", "information"], RECORD.dialogs


def offline_first_run(tmp):
    offline = mock.Mock(side_effect=urllib.error.URLError("offline"))
    seen = len(FakeCapture.opened)
    with app_on("full", "knee_flexion", tmp, "offline", MODEL_DIR=pathlib.Path(tmp) / "empty",
                extra=[(capture.urllib.request, "urlopen", offline)]) as run:
        assert press_start_and_finish(run) == []
        assert [kind for kind, _ in RECORD.dialogs] == ["critical"], RECORD.dialogs
        assert "URLError" in RECORD.dialogs[0][1], RECORD.dialogs
        assert run.window.exercise.isEnabled(), "controls stayed locked after the failure"
        assert len(FakeCapture.opened) == seen, "the camera opened without a model"


def stop_mid_download(tmp):
    """Stop while the model downloads: the thread outlives Stop and must neither crash nor talk."""
    slow = mock.Mock(side_effect=lambda *_a, **_k: SlowDownload())
    with app_on("full", "knee_flexion", tmp, "stall", MODEL_DIR=pathlib.Path(tmp) / "stall-models",
                extra=[(capture.urllib.request, "urlopen", slow)]) as run:
        window = run.window
        QtTest.QTest.mouseClick(window.button, QtCore.Qt.LeftButton)
        assert pump(lambda: slow.called, 10), "the download never started"
        QtTest.QTest.mouseClick(window.button, QtCore.Qt.LeftButton)  # Stop, mid-download
        assert window.worker is None and window._still_stopping(), "Stop waited out the download"
        QtTest.QTest.mouseClick(window.button, QtCore.Qt.LeftButton)  # Start again at once
        assert window.worker is None and "Still stopping" in run.statuses[-1], run.statuses[-1]
        assert pump(lambda: not window._still_stopping(), 15), "the stalled thread never ended"
        QtTest.QTest.qWait(300)
        assert not RECORD.dialogs, f"the let-go thread still spoke: {RECORD.dialogs}"
        assert len(slow.call_args_list) == 1, "a second download started"
    with app_on("full", "knee_flexion", tmp, "after-stall") as run:
        assert len(press_start_and_finish(run)) == 1, "Start did not work again after the stall"


def ratchet_steps_down(tmp):
    with app_on("full", "knee_flexion", tmp, "ratchet", TIERS=("heavy", "lite"),
                BUDGET_MS=0.001, RATCHET_S=60.0) as run:
        files = press_start_and_finish(run)
        assert storage.read_header(files[0])["model"] == "heavy->lite", files
        assert any("too slow for heavy" in s for s in run.statuses), run.statuses[-5:]


def real_video_file(tmp):
    """One clip through real OpenCV decoding, ending at end of file without a camera error."""
    path = pathlib.Path(tmp) / "clip.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 720))
    for frame in CLIPS["full"][:45]:
        out.write(frame)
    out.release()
    cap, frames = REAL_CAPTURE(str(path)), 0
    while cap.read()[0]:
        frames += 1
    cap.release()
    assert frames == 45, f"this OpenCV build wrote {frames}/45 mp4v frames back"
    with app_on(str(path), "knee_flexion", tmp, "file",
                extra=[(capture.cv2, "VideoCapture", guarded_capture)]) as run:
        files = press_start_and_finish(run)
        assert len(files) == 1 and not RECORD.dialogs, (files, RECORD.dialogs)
        assert trace_rows(files[0]) >= 30, trace_rows(files[0])


def missing_clip(tmp):
    with app_on("no-such-clip", "knee_flexion", tmp, "missing") as run:
        assert press_start_and_finish(run) == []
        assert RECORD.dialogs == [("critical", "Could not open no-such-clip.")], RECORD.dialogs
        assert run.window.exercise.isEnabled(), "controls stayed locked after the failure"


SCENARIOS = (upper_body_records, empty_room_saves_nothing, chair_stand_flow, quit_while_recording,
             offline_first_run, stop_mid_download, ratchet_steps_down, real_video_file,
             missing_clip)


def main():
    _app, tmp = sandbox("E2E")
    CLIPS["upper"] = [upper_body_frame()] * 60
    CLIPS["full"] = [sample_frame()] * 240
    CLIPS["empty"] = [np.full((720, 1280, 3), 128, np.uint8)] * 45
    started = time.monotonic()
    try:
        for scenario in SCENARIOS:
            t = time.monotonic()
            try:
                scenario(tmp)
                assert not RECORD.errors, RECORD.errors[0]
            except BaseException:
                save_artifacts(f"e2e-{scenario.__name__}", tmp)
                raise
            print(f"  ok  {scenario.__name__:26} {time.monotonic() - t:5.1f} s")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin"
                                                                 else 2**10)
    left = [w for w in QtWidgets.QApplication.topLevelWidgets() if isinstance(w, rehapose.Main)]
    assert not left, f"{len(left)} windows were never deleted"
    print(f"e2e passed ({len(SCENARIOS)} scenarios, {time.monotonic() - started:.0f} s, "
          f"peak {peak:.0f} MB)")


if __name__ == "__main__":
    main()
