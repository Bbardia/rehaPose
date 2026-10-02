"""Smoke test: real MediaPipe landmarks through the real UI, no camera, no display.

    QT_QPA_PLATFORM=offscreen python test_rehapose.py

Covers the integration that actually breaks: Qt signal payloads, the QImage stride,
pyqtgraph's nan line-breaks, and the results table.
"""
import os
import pathlib
import shutil
import sys
import tempfile

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import analysis
import rehapose
from analysis import JOINTS
from PyQt5 import QtCore, QtWidgets


SAMPLE_URL = ("https://raw.githubusercontent.com/open-mmlab/mmpose/main/"
              "tests/data/coco/000000000785.jpg")


def sample_frame():
    """A real photo of a person - a stick figure does not trigger the detector."""
    import urllib.request
    path = rehapose.MODEL_DIR / "sample_person.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(SAMPLE_URL, path)
    return cv2.resize(cv2.imread(str(path)), (1280, 720))


def main():
    analysis.demo()

    app = QtWidgets.QApplication(sys.argv)
    app.setOrganizationName("rehaPose")
    app.setApplicationName("rehaPoseTest")     # never touch the real preferences
    assert "rehaPoseTest" in rehapose.settings().fileName(), rehapose.settings().fileName()
    # Point the data dir at a scratch folder so the first-run chooser stays silent.
    tmp = tempfile.mkdtemp(prefix="rehapose-test-")
    rehapose.settings().setValue("dataDir", tmp)
    assert rehapose.session_dir() == pathlib.Path(tmp) / "sessions"

    window = rehapose.Main(camera=0)

    # Set up session state the way start() does, without touching the camera.
    window.t0 = 0.0
    window.times = {j: [] for j in JOINTS}
    window.angles = {j: [] for j in JOINTS}
    window.filters = {j: analysis.OneEuro() for j in JOINTS}

    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    path = rehapose.ensure_model("lite")
    landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO, num_poses=1))

    frame = sample_frame()
    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    poses = []
    for i in range(30):
        res = landmarker.detect_for_video(image, (i + 1) * 33)
        world = res.pose_world_landmarks[0] if res.pose_world_landmarks else None
        pixel = res.pose_landmarks[0] if res.pose_landmarks else None
        poses.append((pixel, world))
    landmarker.close()
    detected = sum(1 for _, w in poses if w is not None)
    assert detected > 0, "MediaPipe found no pose in the sample photo"

    # The real gate on real landmarks must produce a real cosine. Which way it decides on
    # this one photo is not the point - analysis.demo() checks that against known yaws.
    pixel, world = poses[0]
    cos = analysis.orientation_cos(pixel, world, frame.shape[1], frame.shape[0])
    assert cos is not None and 0.0 <= cos <= 1.0, cos
    # A refused setup must show the camera but record nothing.
    rehapose.setup_check = lambda *_a, **_k: (False, "Turn side-on to the camera.")
    window.on_frame(frame.copy(), poses[0], 7.0)
    assert window.clock_start is None, "gate let a badly framed session start"
    assert all(not window.angles[j] for j in JOINTS), "gated frames were still recorded"
    assert window.video.pixmap() is not None, "gate should still show the camera"

    # With the setup good, the same frames must record normally.
    rehapose.setup_check = lambda *_a, **_k: (True, "Setup looks good.")
    for i, landmark_pair in enumerate(poses):
        window.t0 = -(i / 30.0)  # advance the clock without sleeping
        window.on_frame(frame.copy(), landmark_pair, 7.0)
        app.processEvents()   # also keeps the QApplication referenced for Qt's lifetime

    for joint in JOINTS:
        assert len(window.angles[joint]) == 30, (joint, len(window.angles[joint]))
    assert len(window.stands) == 30

    before = set(rehapose.session_dir().glob("*.csv"))
    window.show_results()
    assert window.stack.currentIndex() == 1
    assert window.results.item(0, 0) is not None, "results table not populated"
    window._sync()                    # stop() does this; _sync is the only enabler
    assert window.export.isEnabled()
    assert "Saved" in window.status.text(), window.status.text()

    # Every session must be written on stop, without anyone clicking Save.
    written = set(rehapose.session_dir().glob("*.csv")) - before
    assert len(written) == 1, written
    saved = written.pop()
    text = saved.read_text()
    assert "framing_good_pct" in text and "time_s" in text, text[:200]
    assert text.count("\n") > 30, "per-frame trace missing from the autosave"
    saved.unlink()

    # Chair-stand mode: absolute thresholds, counted off whichever knee is more flexed.
    window.exercise.setCurrentIndex(window.exercise.findData("chair_stand_30s"))
    assert window.chair_mode
    window.stands = []
    for _ in range(7):
        window.stands += list(np.linspace(90, 5, 12)) + list(np.linspace(5, 90, 12))
    assert analysis.count_reps(window.stands, lo=analysis.CHAIR_STAND_LO,
                               hi=analysis.CHAIR_STAND_HI) == 7
    window.age.setValue(72)
    window.sex.setCurrentText("female")
    window._snapshot()                # start() does this
    window.stand_count = 7
    line = window.verdict()
    assert "Chair stands: 7" in line and "14" in line, line
    assert "cannot check" in line, "protocol caveat missing from the result"
    for extra in rehapose.session_dir().glob("*.csv"):
        extra.unlink()

    # A frame with no pose at all must not crash and must record a gap.
    window.on_frame(frame.copy(), (None, None), 7.0)
    assert window.angles["left_knee"][-1] is None

    check_shell(window, tmp, frame, poses)
    print(f"smoke test passed ({detected}/30 frames tracked)")


class FakeWorker:
    """Stands in for PoseWorker so stop() can be exercised without a camera."""

    tier = "lite"

    def stop(self):
        pass

    def wait(self, _ms=0):
        return True


def check_shell(window, tmp, frame, poses):
    """The app shell: state machine, menus, history round-trip, junk-session guard."""
    # Modal dialogs block forever offscreen, so silence them for the duration.
    QtWidgets.QMessageBox.critical = staticmethod(lambda *_a, **_k: None)
    QtWidgets.QMessageBox.warning = staticmethod(lambda *_a, **_k: None)
    QtWidgets.QMessageBox.information = staticmethod(lambda *_a, **_k: None)
    # _sync is the only place widgets are enabled. Recording locks the exercise combo
    # and History; idle unlocks them. The old code left the combo disabled forever
    # after a camera error, because on_failed reset the button but not the combo.
    window.worker = FakeWorker()
    window._sync()
    assert not window.exercise.isEnabled() and not window.act_history.isEnabled()
    assert window.button.text() == "Stop" and window.act_run.text() == "Stop"
    # A camera that dies mid-session must keep what was already recorded.
    before = set(rehapose.session_dir().glob("*.csv"))
    window.on_failed("camera exploded")
    assert window.worker is None
    assert window.exercise.isEnabled(), "combo stuck disabled after a failure"
    assert window.button.text() == "Start"
    kept = set(rehapose.session_dir().glob("*.csv")) - before
    assert len(kept) == 1, "camera failure threw the recording away"
    kept.pop().unlink()

    # A frame still queued from a stopped worker must be dropped, not recorded.
    class OldWorker(QtCore.QObject):
        ready = QtCore.pyqtSignal(object, object, float)
    old = OldWorker()
    old.ready.connect(window.on_frame)
    count = len(window.angles["left_knee"])
    old.ready.emit(frame.copy(), poses[0], 7.0)
    assert len(window.angles["left_knee"]) == count, "stale frame was recorded"

    # A save that fails on Stop must say so, not take the process down (exit 134).
    blocker = pathlib.Path(tmp) / "not-a-folder"
    blocker.write_text("")
    rehapose.settings().setValue("dataDir", str(blocker))
    window.show_results()
    assert "NOT SAVED" in window.status.text(), window.status.text()
    rehapose.settings().setValue("dataDir", tmp)

    # Stopping with nothing ever measured must not write an all-zeros session.
    recorded, window.times = window.times, {j: [] for j in JOINTS}
    window.summaries = {}
    window.worker = FakeWorker()
    before = set(rehapose.session_dir().glob("*.csv"))
    window.stop()
    assert window.worker is None
    assert set(rehapose.session_dir().glob("*.csv")) == before, "junk session written"
    window.times = recorded

    # Exercise keys, not display text, are what land in the file and drive History.
    window.exercise.setCurrentIndex(window.exercise.findData("knee_flexion"))
    window._snapshot()
    # Changing the combo after Stop must not relabel what was recorded.
    window.exercise.setCurrentIndex(window.exercise.findData("heel_slides"))
    window.summaries = {j: analysis.summarize(window.angles[j]) for j in JOINTS}
    saved = window.autosave()
    assert saved.name.endswith("-knee_flexion.csv"), saved.name
    window.exercise.setCurrentIndex(window.exercise.findData("knee_flexion"))
    head = rehapose.read_header(saved)
    assert head["exercise"] == "knee_flexion", head
    assert "duration_s" in head and "best_rom" in head, head

    # History must read what autosave wrote, and round-trip it back into the results.
    window.show_history()
    assert window.pages.currentIndex() == 1
    assert window.sessions.rowCount() == 1, window.sessions.rowCount()
    assert window.sessions.item(0, 1).text() == "Knee flexion / extension"
    window.open_stored(window.sessions.item(0, 0))
    assert window.pages.currentIndex() == 0 and window.stack.currentIndex() == 1
    # Viewing a stored session must not let Save a Copy write the live session over it.
    assert window.viewing_stored == saved
    assert not window.export.isEnabled() and not window.act_save.isEnabled()

    # Empty state: zero sessions must say so rather than render a blank table.
    saved.unlink()
    window.show_history()
    assert window.sessions.rowCount() == 0
    assert "No sessions yet" in window.caption.text()

    # Settings round-trip, so the app reopens the way it was left.
    window.age.setValue(81)
    window.cue.setChecked(False)
    window.save_settings()
    assert rehapose.settings().value("age", type=int) == 81
    assert rehapose.settings().value("exercise", type=str) == "knee_flexion"
    assert rehapose.settings().value("beep", True, type=bool) is False

    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
