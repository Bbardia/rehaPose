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


# Pinned to a commit: "main" of an unmaintained repo can move or vanish under the test.
SAMPLE_URL = ("https://raw.githubusercontent.com/open-mmlab/mmpose/"
              "ec2f372f002d1d534ea01a13033d09f5483256db/tests/data/coco/000000000785.jpg")


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
    # Modal dialogs block forever offscreen, so a regression that opens one would hang
    # the run instead of failing it. Silence them before anything can open one.
    QtWidgets.QMessageBox.critical = staticmethod(lambda *_a, **_k: None)
    QtWidgets.QMessageBox.warning = staticmethod(lambda *_a, **_k: None)
    QtWidgets.QMessageBox.information = staticmethod(lambda *_a, **_k: None)
    # Point the data dir at a scratch folder so the first-run chooser stays silent.
    tmp = tempfile.mkdtemp(prefix="rehapose-test-")
    try:
        run(app, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run(app, tmp):
    rehapose.settings().clear()           # last run's saved exercise must not leak in
    rehapose.settings().setValue("dataDir", tmp)
    assert rehapose.session_dir() == pathlib.Path(tmp) / "sessions"

    window = rehapose.Main(camera=0)
    # Nobody has entered an age or sex yet, so neither may default to a real value.
    assert window.age.value() == rehapose.AGE_UNSET, window.age.value()
    assert window.sex.currentText() == rehapose.SEX_UNSET, window.sex.currentText()
    window.exercise.setCurrentIndex(window.exercise.findData("knee_flexion"))
    window._snapshot()

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
    pixel, world = poses[-1]
    cos = analysis.orientation_cos(pixel, world, frame.shape[1], frame.shape[0])
    assert cos is not None and 0.0 <= cos <= 1.0, cos
    # Mirroring the photo turns the same body the other way; the gate must read nearly
    # the same turn. On heavy, the default tier, shoulders alone moved 0.44 -> 0.81 for
    # this photo; shoulders and hips averaged move 0.54 -> 0.65.
    turns = []
    for image_bgr in (frame, cv2.flip(frame, 1)):
        heavy = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(rehapose.ensure_model(
                    "heavy"))), running_mode=vision.RunningMode.VIDEO, num_poses=1))
        still = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        for i in range(15):
            res = heavy.detect_for_video(still, (i + 1) * 33)
        heavy.close()
        turns.append(analysis.orientation_cos(res.pose_landmarks[0],
                                              res.pose_world_landmarks[0],
                                              frame.shape[1], frame.shape[0]))
    assert abs(turns[0] - turns[1]) < 0.15, turns
    # A refused setup must show the camera but record nothing.
    rehapose.setup_check = lambda *_a, **_k: (False, "Turn side-on to the camera.")
    window.on_frame(frame.copy(), poses[0], 7.0)
    assert window.clock_start is None, "gate let a badly framed session start"
    assert all(not window.angles[j] for j in JOINTS), "gated frames were still recorded"
    assert window.video.pixmap() is not None, "gate should still show the camera"

    # With the setup good, the same frames must record normally - once it has held for
    # SETUP_HOLD frames, so a single lucky frame cannot start a session.
    rehapose.setup_check = lambda *_a, **_k: (True, "Setup looks good.")
    for _ in range(rehapose.SETUP_HOLD - 1):
        window.on_frame(frame.copy(), poses[0], 7.0)
    assert window.clock_start is None, "one good frame short of the hold, and it started"
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
    window.time_called = True
    line = window.verdict()
    assert "Chair stands: 7" in line and "14" in line, line
    assert "cannot check" in line, "protocol caveat missing from the result"
    # Stopped by hand before 30 s: the count is not a score, so no norm beside it.
    window.time_called = False
    line = window.verdict()
    assert "not a 30-second score" in line and "14" not in line, line
    for extra in rehapose.session_dir().glob("*.csv"):
        extra.unlink()

    check_chair_stand(window, app, frame, poses)

    # A frame with no pose at all must not crash and must record a gap.
    window.on_frame(frame.copy(), (None, None), 7.0)
    assert window.angles["left_knee"][-1] is None

    check_shell(window, tmp, frame, poses)
    print(f"smoke test passed ({detected}/30 frames tracked)")


def check_chair_stand(window, app, frame, poses):
    """Countdown, then measure, then time is called with the final-stand rule applied."""
    import time
    kept = window.times, window.angles, window.stands
    window._snapshot()
    window.times = {j: [] for j in JOINTS}
    window.angles = {j: [] for j in JOINTS}
    window.stands, window.stand_count, window.clock_start = [], 0, None
    window.counted, window.time_called, window.final_counted = None, False, False
    window.frames = window.setup_ok = 0
    window.good_run = rehapose.SETUP_HOLD - 1
    window.on_frame(frame.copy(), poses[0], 7.0)
    assert window.clock_start > time.perf_counter(), "no countdown before a scored test"
    assert "starting in 3" in window.status.text(), window.status.text()
    assert window.counted == 3 and not window.measured, "countdown frames were recorded"
    assert window.frames == 0, "countdown frames counted towards framing"

    window.clock_start = window.t0 = time.perf_counter() - 1.0      # countdown over
    window.on_frame(frame.copy(), poses[0], 7.0)
    assert window.measured and window.counted == 0

    # Three full stands and a rise past halfway, then time is called.
    window.stands = []
    for _ in range(3):
        window.stands += list(np.linspace(90, 5, 12)) + list(np.linspace(5, 90, 12))
    window.stands += list(np.linspace(90, 40, 6))
    window.worker = FakeWorker()
    window.t0 = time.perf_counter() - analysis.CHAIR_STAND_SECONDS - 1.0
    window.on_frame(frame.copy(), (None, None), 7.0)   # a gap, so the rise stays last
    app.processEvents()
    assert window.worker is None, "time ran out but the session kept recording"
    assert window.stand_count == 4 and window.final_counted, window.stand_count
    assert "more than halfway" in window.status.text(), window.status.text()
    saved = sorted(rehapose.session_dir().glob("*-chair_stand_30s.csv"))[-1]
    head = rehapose.read_header(saved)
    assert head["stands"] == "4" and head["complete"] == "yes", head

    # And History brings the stand count back, not just the joint table.
    window.show_history()
    window.open_stored(window.sessions.item(0, 0))
    assert "Chair stands: 4" in window.status.text(), window.status.text()
    for extra in rehapose.session_dir().glob("*.csv"):
        extra.unlink()

    # Stopped by hand: the file, History and the stored view must all say so.
    window.viewing_stored, window.time_called = None, False
    saved = window.autosave()
    assert rehapose.read_header(saved)["complete"] == "no"
    window.show_history()
    assert "(stopped early)" in window.sessions.item(0, 4).text()
    window.open_stored(window.sessions.item(0, 0))
    assert "stopped before 30 s" in window.status.text(), window.status.text()
    saved.unlink()
    window.times, window.angles, window.stands = kept


class FakeWorker:
    """Stands in for PoseWorker so stop() can be exercised without a camera."""

    tier_log = ["lite"]

    def stop(self):
        pass

    def wait(self, _ms=0):
        return True


real_session_dir = rehapose.session_dir


def check_shell(window, tmp, frame, poses):
    """The app shell: state machine, menus, history round-trip, junk-session guard."""
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
    assert window.unsaved
    # While unsaved, nothing may quietly replace the only copy on screen.
    box = QtWidgets.QMessageBox
    box.question = staticmethod(lambda *_a, **_k: box.Cancel)
    window.start()
    assert window.worker is None, "Start discarded an unsaved session"
    warned = []
    box.warning = staticmethod(lambda *a, **_k: warned.append(a[2]))
    window.history_files = [blocker]
    window.open_stored(QtWidgets.QTableWidgetItem())
    assert warned and "not saved" in warned[0], warned
    box.warning = staticmethod(lambda *_a, **_k: None)
    # Once the folder is back, opening History retries the save by itself.
    rehapose.settings().setValue("dataDir", tmp)
    before = set(rehapose.session_dir().glob("*.csv"))
    window.show_history()
    assert not window.unsaved and "on retry" in window.status.text(), window.status.text()
    retried = set(rehapose.session_dir().glob("*.csv")) - before
    assert len(retried) == 1
    retried.pop().unlink()
    window.pages.setCurrentIndex(0)

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
    assert head["best_joint"] in JOINTS, head
    assert head["app_version"] == rehapose.VERSION and head["mediapipe"] == "1.0.0", head
    assert head["model"] == "lite", head            # from FakeWorker, via _detach

    # The file is the user's only copy: what it reads back must be what was shown.
    stored = rehapose.stored_summaries(saved)
    assert set(stored) == set(JOINTS), stored.keys()
    for joint, live in window.summaries.items():
        for field in ("rom", "peak", "min", "coverage"):
            assert abs(stored[joint][field] - live[field]) <= 0.05, (joint, field)
        assert stored[joint]["reps"] == live["reps"], joint

    # Two stops in the same second must give two files, not one overwritten.
    twin = window.autosave()
    assert twin != saved and twin.exists() and saved.exists(), (saved, twin)
    twin.unlink()

    # Excel and Numbers pad blank rows with commas; the blocks must still split.
    padded = saved.with_name("20200104-000000-knee_flexion.csv")
    raw = saved.read_bytes().replace(b"\r\n\r\n", b"\r\n,,,,,\r\n")
    assert raw.count(b"\r\n,,,,,\r\n") == 2, "test did not pad anything"
    padded.write_bytes(raw)
    assert rehapose.stored_summaries(padded) == stored, "padded blank rows broke the reader"
    padded.unlink()

    # A spreadsheet re-save in a legacy encoding, and a file that is not a session at
    # all, must each cost one row - not the whole History page.
    mangled = saved.with_name("20200101-000000-knee_flexion.csv")
    mangled.write_bytes(saved.read_text(encoding="utf-8-sig").encode("cp1252"))
    junk = saved.with_name("20200102-000000-other.csv")
    junk.write_bytes(b"\x00\xff garbage\r\n\r\njoint\r\nleft_knee,notanumber\r\n")
    unopenable = saved.with_name("20200103-000000-dir.csv")
    unopenable.mkdir()                                 # open() raises IsADirectoryError

    # History must read what autosave wrote, and round-trip it back into the results.
    window.show_history()
    assert window.pages.currentIndex() == 1
    assert window.sessions.rowCount() == 4, window.sessions.rowCount()
    assert window.sessions.item(1, 2).text() == "(unreadable)"
    assert window.sessions.item(0, 2).text() == "Knee flexion / extension"
    assert head["best_joint"].replace("_", " ") in window.sessions.item(0, 4).text()
    # The degree sign decodes as U+FFFD; the number, which is what matters, survives.
    assert any(c.isdigit() for c in window.sessions.item(3, 4).text()), "cp1252 row lost"
    viewing = window.viewing_stored
    window.open_stored(window.sessions.item(2, 0))      # junk: a warning, not a crash
    assert window.viewing_stored == viewing, "a refused file still took over the view"
    mangled.unlink()
    junk.unlink()
    unopenable.rmdir()
    window.show_history()
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

    # Age and sex follow the person code, so one patient's norm never lands on the next.
    window.person.setText("AB")
    window.on_person_changed()            # a new code asks for age and sex again...
    assert window.age.value() == rehapose.AGE_UNSET
    assert window.sex.currentText() == rehapose.SEX_UNSET
    window.age.setValue(72)               # ...which are entered after it
    window.sex.setCurrentText("male")
    window._snapshot()
    window.remember_person()
    window.person.setText("CD")
    window.on_person_changed()
    assert window.age.value() == rehapose.AGE_UNSET, "new person inherited an age"
    assert window.sex.currentText() == rehapose.SEX_UNSET, "new person inherited a sex"
    window._snapshot()
    assert window.recorded["age"] is None and window.recorded["sex"] is None
    window.recorded["chair"], window.time_called = True, True
    assert "not entered" in window.verdict(), window.verdict()
    saved = window.autosave()
    head = rehapose.read_header(saved)
    assert head["person"] == "CD" and head["age"] == "" and head["sex"] == "", head
    window.show_history()
    assert window.sessions.item(0, 1).text() == "CD"
    saved.unlink()
    window.person.setText("AB")
    window.on_person_changed()
    assert window.age.value() == 72, "known person's age was not restored"
    assert window.sex.currentText() == "male", "known person's sex was not restored"

    # Legacy sessions move across whole, leaving no half-copied file behind.
    legacy = pathlib.Path(tmp) / "legacy"
    legacy.mkdir()
    (legacy / "20250101-000000-free.csv").write_text("exercise,free\r\n")
    real_legacy, rehapose.LEGACY_SESSIONS = rehapose.LEGACY_SESSIONS, legacy
    target = pathlib.Path(tmp) / "migrated"
    assert rehapose.migrate_legacy(target) == 1
    assert [f.name for f in target.iterdir()] == ["20250101-000000-free.csv"]
    assert not list(legacy.iterdir()), "legacy original left behind"
    rehapose.LEGACY_SESSIONS = real_legacy

    # The sessions folder went missing and the picker was cancelled: keep the old
    # folder rather than silently starting a second one in Documents.
    missing = str(pathlib.Path(tmp) / "unplugged-drive")
    rehapose.settings().setValue("dataDir", missing)
    picker = QtWidgets.QFileDialog.getExistingDirectory
    QtWidgets.QFileDialog.getExistingDirectory = staticmethod(lambda *_a, **_k: "")
    rehapose.session_dir = lambda: None          # the drive is not there
    try:
        assert rehapose.choose_data_dir(window) is None
        assert rehapose.settings().value("dataDir", type=str) == missing
    finally:
        QtWidgets.QFileDialog.getExistingDirectory = picker
        rehapose.session_dir = real_session_dir
        rehapose.settings().setValue("dataDir", tmp)

    # Settings round-trip, so the app reopens the way it was left.
    window.age.setValue(81)
    window.cue.setChecked(False)
    window.save_settings()
    assert rehapose.settings().value("age", type=int) == 81
    assert rehapose.settings().value("exercise", type=str) == "knee_flexion"
    assert rehapose.settings().value("beep", True, type=bool) is False


if __name__ == "__main__":
    main()
