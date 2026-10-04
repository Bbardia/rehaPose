"""Headless UI smoke test: QT_QPA_PLATFORM=offscreen python test_rehapose.py"""
import os
import pathlib
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import analysis
import capture
import rehapose
import storage
from analysis import JOINTS
from PyQt5 import QtCore, QtGui, QtWidgets


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
    # Modal dialogs block forever offscreen: stub them so a regression fails, not hangs.
    QtWidgets.QMessageBox.critical = staticmethod(lambda *_a, **_k: None)
    QtWidgets.QMessageBox.warning = staticmethod(lambda *_a, **_k: None)
    QtWidgets.QMessageBox.information = staticmethod(lambda *_a, **_k: None)
    QtWidgets.QFileDialog.getExistingDirectory = staticmethod(lambda *_a, **_k: "")
    QtWidgets.QFileDialog.getSaveFileName = staticmethod(lambda *_a, **_k: ("", ""))
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
    assert window.age.value() == rehapose.AGE_UNSET, window.age.value()
    assert window.sex.currentText() == rehapose.SEX_UNSET, window.sex.currentText()
    window.exercise.setCurrentIndex(window.exercise.findData("knee_flexion"))
    window._snapshot()

    window.t0 = 0.0
    window.times = {j: [] for j in JOINTS}
    window.angles = {j: [] for j in JOINTS}
    window.filters = {j: analysis.OneEuro() for j in JOINTS}

    frame, poses = track_sample()
    detected = sum(1 for _, w in poses if w is not None)
    assert detected > 0, "MediaPipe found no pose in the sample photo"

    check_orientation(frame, poses)
    check_overlay()
    check_setup_gate(window, app, frame, poses)
    check_autosave(window)
    check_chair_verdict(window)
    check_chair_stand(window, app, frame, poses)

    window.on_frame(frame.copy(), (None, None), 7.0)
    assert window.angles["left_knee"][-1] is None

    check_shell(window, tmp, frame, poses)
    print(f"smoke test passed ({detected}/30 frames tracked)")


def track_sample():
    """The sample photo and 30 (pixel, world) landmark pairs from the lite model."""
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    path = capture.ensure_model("lite")
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
    return frame, poses


def check_orientation(frame, poses):
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    pixel, world = poses[-1]
    cos = analysis.orientation_cos(pixel, world, frame.shape[1], frame.shape[0])
    assert cos is not None and 0.0 <= cos <= 1.0, cos
    # The mirrored photo must read nearly the same turn on heavy (shoulders alone failed).
    turns = []
    for image_bgr in (frame, cv2.flip(frame, 1)):
        heavy = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(capture.ensure_model(
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


def check_overlay():
    pixel, _ = analysis._body(0)
    for i in (25, 26, 27, 28):
        pixel[i] = analysis._P(pixel[i].x, pixel[i].y, 0, v=0.1)
    drawn = rehapose.draw_overlay(np.zeros((720, 1280, 3), np.uint8), pixel)
    assert drawn[int(pixel[11].y * 720), int(pixel[11].x * 1280)].any(), "shoulder not drawn"
    ankle = int(pixel[27].y * 720), int(pixel[27].x * 1280)
    assert not drawn[ankle[0] - 3:ankle[0] + 4, ankle[1] - 3:ankle[1] + 4].any(), "phantom leg"


def check_labels(window):
    pixel = [analysis._P(0.5, 0.5, 0, v=0.0) for _ in range(33)]
    pixel[25] = analysis._P(0.25, 0.5, 0)
    shown, chair = [], window.recorded["chair"]
    window.recorded["chair"] = False
    window.show_frame = shown.append
    window.show_live(np.zeros((720, 1280, 3), np.uint8), pixel, {"left_knee": 42.0})
    del window.show_frame
    expected = cv2.flip(rehapose.draw_overlay(np.zeros((720, 1280, 3), np.uint8), pixel), 1)
    cv2.putText(expected, "42", (1279 - 320 + 8, 360 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    assert np.array_equal(shown[0], expected), "angle label mirrored or misplaced on the video"
    window.recorded["chair"] = chair


def check_setup_gate(window, app, frame, poses):
    window.on_frame(frame.copy(), (None, None), 7.0)
    assert window.clock_start is None, "the clock started with nobody in view"
    assert all(not window.angles[j] for j in JOINTS), "frames without a person were recorded"
    assert window.video.pixmap() is not None, "gate should still show the camera"
    assert "No person" in window.status.text(), window.status.text()

    # Bad framing is advice only, but the start needs SETUP_HOLD CONSECUTIVE frames in view.
    rehapose.setup_check = lambda *_a, **_k: (False, "Turn side-on to the camera.")
    hold = rehapose.SETUP_HOLD - 1
    drawn = []
    real_draw, rehapose.draw_overlay = rehapose.draw_overlay, lambda f, *_a: drawn.append(1) or f
    for pair in [poses[0]] * hold + [(None, None)] + [poses[0]] * hold:
        window.on_frame(frame.copy(), pair, 7.0)
    rehapose.draw_overlay = real_draw
    assert window.clock_start is None, "one frame short of the hold, and it started"
    assert len(drawn) == 2 * hold, "no skeleton while waiting for the clock"
    for i, landmark_pair in enumerate(poses):
        window.t0 = -(i / 30.0)  # advance the clock without sleeping
        window.on_frame(frame.copy(), landmark_pair, 7.0)
        app.processEvents()   # also keeps the QApplication referenced for Qt's lifetime

    for joint in JOINTS:
        assert len(window.angles[joint]) == 30, (joint, len(window.angles[joint]))
    assert len(window.stands) == 30
    assert "Turn side-on" in window.status.text(), "framing advice not shown while recording"
    assert (window.frames, window.setup_ok) == (30, 0), "badly framed frames counted as good"
    rehapose.setup_check = lambda *_a, **_k: (True, "Setup looks good.")


def check_autosave(window):
    before = set(rehapose.session_dir().glob("*.csv"))
    window.show_results()
    assert window.stack.currentIndex() == 1
    assert window.results.item(0, 0) is not None, "results table not populated"
    window._sync()                    # stop() does this; _sync is the only enabler
    assert window.export.isEnabled()
    assert "Saved" in window.status.text(), window.status.text()

    written = set(rehapose.session_dir().glob("*.csv")) - before
    assert len(written) == 1, written
    saved = written.pop()
    text = saved.read_text()
    assert "framing_good_pct" in text and "time_s" in text, text[:200]
    assert text.count("\n") > 30, "per-frame trace missing from the autosave"
    saved.unlink()


def check_chair_verdict(window):
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
    assert "Chair stands: 7" in line and "female aged 72: 14." in line, line
    assert "cannot check" in line, "protocol caveat missing from the result"
    window.time_called = False
    line = window.verdict()
    assert "not a 30-second score" in line and "Reference for" not in line, line
    for extra in rehapose.session_dir().glob("*.csv"):
        extra.unlink()


def check_chair_stand(window, app, frame, poses):
    """Countdown, then measure, then time is called with the final-stand rule applied."""
    kept = window.times, window.angles, window.stands
    check_countdown(window, frame, poses)
    check_time_called(window, app, frame)
    check_stopped_early(window)
    window.times, window.angles, window.stands = kept


def check_countdown(window, frame, poses):
    window._snapshot()
    window.times = {j: [] for j in JOINTS}
    window.angles = {j: [] for j in JOINTS}
    window.stands, window.stand_count, window.clock_start = [], 0, None
    window.counted, window.time_called, window.final_counted = None, False, False
    window.frames = window.setup_ok = window.good_run = 0
    real_in_view, rehapose.in_view = rehapose.in_view, lambda *_a, **_k: []
    for _ in range(rehapose.SETUP_HOLD + 2):
        window.on_frame(frame.copy(), poses[0], 7.0)
    rehapose.in_view = real_in_view
    assert window.clock_start is None, "the chair stand started without a knee in view"
    window.good_run = rehapose.SETUP_HOLD - 1
    window.on_frame(frame.copy(), poses[0], 7.0)
    assert window.clock_start > time.perf_counter(), "no countdown before a scored test"
    assert "starting in 3" in window.status.text(), window.status.text()
    assert window.counted == 3 and not window.measured, "countdown frames were recorded"
    assert window.frames == 0, "countdown frames counted towards framing"

    window.clock_start = window.t0 = time.perf_counter() - 1.0      # countdown over
    window.on_frame(frame.copy(), poses[0], 7.0)
    assert window.measured and window.counted == 0


def check_time_called(window, app, frame):
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
    head = storage.read_header(saved)
    assert head["stands"] == "4" and head["complete"] == "yes", head

    window.show_history()
    window.open_stored(window.sessions.item(0, 0))
    assert "Chair stands: 4" in window.status.text(), window.status.text()
    for extra in rehapose.session_dir().glob("*.csv"):
        extra.unlink()


def check_stopped_early(window):
    window.viewing_stored, window.time_called = None, False
    saved = window.autosave()
    assert storage.read_header(saved)["complete"] == "no"
    window.show_history()
    assert "(stopped early)" in window.sessions.item(0, 4).text()
    window.open_stored(window.sessions.item(0, 0))
    assert "stopped before 30 s" in window.status.text(), window.status.text()
    saved.unlink()


class FakeWorker:
    """Stands in for PoseWorker so stop() can be exercised without a camera."""

    tier_log = ["heavy", "lite"]

    def stop(self):
        pass

    def wait(self, _ms=0):
        return True


real_session_dir = storage.session_dir


def check_shell(window, tmp, frame, poses):
    """The app shell: state machine, menus, history round-trip, junk-session guard."""
    check_sync(window)
    check_labels(window)
    check_stale_frame(window, frame, poses)
    blocker = check_failed_save(window, tmp)
    check_unsaved_guard(window, tmp, blocker)
    check_still_stopping(window)
    check_save_retry(window, tmp)
    check_no_junk_session(window)
    check_clip_end(window)
    saved, head = check_exercise_keys(window)
    stored = check_round_trip(window, saved)
    check_half_write(window, saved)
    check_twin_stops(window, saved)
    check_padded_rows(saved, stored)
    check_history(window, saved, head)
    check_empty_history(window, saved)
    check_person_reset(window)
    check_legacy_migration(tmp)
    check_missing_folder(window, tmp)
    check_settings(window)


def check_sync(window):
    window.worker = FakeWorker()
    window._sync()
    assert not window.exercise.isEnabled() and not window.act_history.isEnabled()
    assert window.button.text() == "Stop" and window.act_run.text() == "Stop"
    before = set(rehapose.session_dir().glob("*.csv"))
    window.on_failed("camera exploded")
    assert window.worker is None
    assert window.exercise.isEnabled(), "combo stuck disabled after a failure"
    assert window.button.text() == "Start"
    kept = set(rehapose.session_dir().glob("*.csv")) - before
    assert len(kept) == 1, "camera failure threw the recording away"
    kept.pop().unlink()


def check_stale_frame(window, frame, poses):
    class Queued(QtCore.QThread):
        ready = QtCore.pyqtSignal(object, object, float)
        tier_log = ["lite"]

        def stop(self):
            pass

        def run(self):
            for _ in range(3):
                self.ready.emit(frame.copy(), poses[0], 7.0)

    worker = Queued()
    worker.ready.connect(window.on_frame)
    window.worker = worker
    worker.start()
    worker.wait()  # three frames queued and the thread done, as at a real Stop
    count = len(window.angles["left_knee"])
    window._detach()
    del worker  # only the window may keep it alive, as in the app
    QtWidgets.QApplication.processEvents()
    assert len(window.angles["left_knee"]) == count, "stale frame was recorded"
    window.retiring = None


def check_failed_save(window, tmp):
    blocker = pathlib.Path(tmp) / "not-a-folder"
    blocker.write_text("")
    rehapose.settings().setValue("dataDir", str(blocker))
    window.show_results()
    assert "NOT SAVED" in window.status.text(), window.status.text()
    assert window.unsaved
    return blocker


def check_unsaved_guard(window, tmp, blocker):
    box = QtWidgets.QMessageBox
    box.question = staticmethod(lambda *_a, **_k: box.Cancel)
    window.start()
    assert window.worker is None, "Start discarded an unsaved session"
    closing = QtGui.QCloseEvent()
    window.closeEvent(closing)
    assert not closing.isAccepted(), "Quit discarded an unsaved session"
    # Save chosen, then the save dialog cancelled: still unsaved, still refused.
    box.question = staticmethod(lambda *_a, **_k: box.Save)
    assert not window.confirm_discard() and window.unsaved
    copy = pathlib.Path(tmp) / "copy.csv"
    dialog = QtWidgets.QFileDialog.getSaveFileName
    QtWidgets.QFileDialog.getSaveFileName = staticmethod(lambda *_a, **_k: (str(copy), ""))
    assert window.confirm_discard() and not window.unsaved and copy.exists()
    QtWidgets.QFileDialog.getSaveFileName = dialog
    copy.unlink()
    window.unsaved = True
    box.question = staticmethod(lambda *_a, **_k: box.Discard)
    assert window.confirm_discard() and not window.unsaved
    window.unsaved = True                     # for the History refusal and retry below
    warned = []
    box.warning = staticmethod(lambda *a, **_k: warned.append(a[2]))
    window.history_files = [blocker]
    window.open_stored(QtWidgets.QTableWidgetItem())
    assert warned and "not saved" in warned[0], warned
    box.warning = staticmethod(lambda *_a, **_k: None)


def check_still_stopping(window):
    class StillRunning(QtCore.QObject):
        def isRunning(self):
            return True
    window.unsaved, window.retiring = False, StillRunning()
    window.start()
    assert window.worker is None and "Still stopping" in window.status.text()
    window.unsaved, window.retiring = True, None


def check_save_retry(window, tmp):
    rehapose.settings().setValue("dataDir", tmp)
    before = set(rehapose.session_dir().glob("*.csv"))
    window.show_history()
    assert not window.unsaved and "on retry" in window.status.text(), window.status.text()
    retried = set(rehapose.session_dir().glob("*.csv")) - before
    assert len(retried) == 1
    retried.pop().unlink()
    window.pages.setCurrentIndex(0)


def check_no_junk_session(window):
    recorded, window.times = window.times, {j: [] for j in JOINTS}
    window.summaries = {}
    before = set(rehapose.session_dir().glob("*.csv"))
    for chair, said in ((False, "you were never in view"), (True, "your knees were never")):
        window.worker, window.clock_start = FakeWorker(), None
        window.recorded["chair"] = chair
        window.stop()
        assert window.worker is None
        assert said in window.status.text(), window.status.text()
    assert set(rehapose.session_dir().glob("*.csv")) == before, "junk session written"
    window.recorded["chair"] = False
    window.times = recorded


def check_clip_end(window):
    window.worker = FakeWorker()
    before = set(rehapose.session_dir().glob("*.csv"))
    window.on_ended()  # a clip ran out: the same as pressing Stop
    assert window.worker is None, "the end of a clip did not stop the session"
    added = set(rehapose.session_dir().glob("*.csv")) - before
    assert len(added) == 1, "the end of a clip did not save the session"
    added.pop().unlink()


def check_exercise_keys(window):
    window.exercise.setCurrentIndex(window.exercise.findData("knee_flexion"))
    window._snapshot()
    window.exercise.setCurrentIndex(window.exercise.findData("heel_slides"))
    window.summaries = {j: analysis.summarize(window.angles[j]) for j in JOINTS}
    saved = window.autosave()
    assert saved.name.endswith("-knee_flexion.csv"), saved.name
    window.exercise.setCurrentIndex(window.exercise.findData("knee_flexion"))
    head = storage.read_header(saved)
    assert head["exercise"] == "knee_flexion", head
    assert "duration_s" in head and "best_rom" in head, head
    assert head["best_joint"] in JOINTS, head
    assert head["app_version"] == rehapose.VERSION and head["mediapipe"] == "1.0.0", head
    assert head["model"] == "heavy->lite", head     # the whole tier history, via _detach
    return saved, head


def check_round_trip(window, saved):
    stored = storage.stored_summaries(saved)
    assert set(stored) == set(JOINTS), stored.keys()
    for joint, live in window.summaries.items():
        for field in ("rom", "peak", "min", "coverage"):
            assert abs(stored[joint][field] - live[field]) <= 0.05, (joint, field)
        assert stored[joint]["reps"] == live["reps"], joint
    return stored


def check_half_write(window, saved):
    def half_write(part, *_a):
        pathlib.Path(part).write_text("exercise,knee_flexion\r\n")
        raise OSError(28, "No space left on device")
    real_write, storage._write = storage._write, half_write
    doomed = saved.with_name("20200105-000000-knee_flexion.csv")
    try:
        storage.write_session(doomed, [], window.summaries, window.times, window.angles)
        raise AssertionError("a failed write was reported as saved")
    except OSError:
        pass
    finally:
        storage._write = real_write
    assert not doomed.exists() and not list(saved.parent.glob("*.part")), "truncated file"


def check_twin_stops(window, saved):
    twin = window.autosave()
    assert twin != saved and twin.exists() and saved.exists(), (saved, twin)
    twin.unlink()


def check_padded_rows(saved, stored):
    padded = saved.with_name("20200104-000000-knee_flexion.csv")
    raw = saved.read_bytes().replace(b"\r\n\r\n", b"\r\n,,,,,\r\n")
    assert raw.count(b"\r\n,,,,,\r\n") == 2, "test did not pad anything"
    padded.write_bytes(raw)
    assert storage.stored_summaries(padded) == stored, "padded blank rows broke the reader"
    padded.unlink()


def check_history(window, saved, head):
    mangled = saved.with_name("20200101-000000-knee_flexion.csv")
    mangled.write_bytes(saved.read_text(encoding="utf-8-sig").encode("cp1252"))
    junk = saved.with_name("20200102-000000-other.csv")
    junk.write_bytes(b"\x00\xff garbage\r\n\r\njoint\r\nleft_knee,notanumber\r\n")
    unopenable = saved.with_name("20200103-000000-dir.csv")
    unopenable.mkdir()                                 # open() raises IsADirectoryError

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
    assert window.viewing_stored == saved
    assert not window.export.isEnabled() and not window.act_save.isEnabled()


def check_empty_history(window, saved):
    saved.unlink()
    window.show_history()
    assert window.sessions.rowCount() == 0
    assert "No sessions yet" in window.caption.text()


def check_person_reset(window):
    window.person.setText("AB")
    window.on_person_changed()            # a new code asks for age and sex again...
    assert window.age.value() == rehapose.AGE_UNSET
    assert window.sex.currentText() == rehapose.SEX_UNSET
    window.age.setValue(72)               # ...which are entered after it
    window.sex.setCurrentText("male")
    window._snapshot()
    window.remember_person()
    window.person.setText("CD")
    window.person.textEdited.emit("CD")   # typing a code resets at once, not on focus-out
    assert window.age.value() == rehapose.AGE_UNSET, "new person inherited an age"
    assert window.sex.currentText() == rehapose.SEX_UNSET, "new person inherited a sex"
    window._snapshot()
    assert window.recorded["age"] is None and window.recorded["sex"] is None
    window.recorded["chair"], window.time_called = True, True
    assert "not entered" in window.verdict(), window.verdict()
    saved = window.autosave()
    head = storage.read_header(saved)
    assert head["person"] == "CD" and head["age"] == "" and head["sex"] == "", head
    window.show_history()
    assert window.sessions.item(0, 1).text() == "CD"
    saved.unlink()
    window.person.setText("AB")
    window.on_person_changed()
    assert window.age.value() == 72, "known person's age was not restored"
    assert window.sex.currentText() == "male", "known person's sex was not restored"


def check_legacy_migration(tmp):
    legacy = pathlib.Path(tmp) / "legacy"
    legacy.mkdir()
    (legacy / "20250101-000000-free.csv").write_text("exercise,free\r\n")
    real_legacy, storage.LEGACY_SESSIONS = storage.LEGACY_SESSIONS, legacy
    target = pathlib.Path(tmp) / "migrated"
    assert storage.migrate_legacy(target) == 1
    assert [f.name for f in target.iterdir()] == ["20250101-000000-free.csv"]
    assert not list(legacy.iterdir()), "legacy original left behind"
    (legacy / "20250102-000000-free.csv").write_text("exercise,free\r\n" * 100)
    def half_copy(src, dst):
        pathlib.Path(dst).write_bytes(pathlib.Path(src).read_bytes()[:10])
        raise OSError(28, "No space left on device")
    real_copy, storage.shutil.copy2 = storage.shutil.copy2, half_copy
    try:
        storage.migrate_legacy(target)
        raise AssertionError("a failed copy was reported as a migration")
    except OSError:
        pass
    finally:
        storage.shutil.copy2 = real_copy
    assert not (target / "20250102-000000-free.csv").exists(), "truncated file kept"
    assert not list(target.glob("*.part")) and (legacy / "20250102-000000-free.csv").exists()
    storage.LEGACY_SESSIONS = real_legacy


def check_missing_folder(window, tmp):
    missing = str(pathlib.Path(tmp) / "unplugged-drive")
    rehapose.settings().setValue("dataDir", missing)
    picker = QtWidgets.QFileDialog.getExistingDirectory
    QtWidgets.QFileDialog.getExistingDirectory = staticmethod(lambda *_a, **_k: "")
    storage.session_dir = lambda: None          # the drive is not there
    try:
        assert storage.choose_data_dir(window) is None
        assert rehapose.settings().value("dataDir", type=str) == missing
    finally:
        QtWidgets.QFileDialog.getExistingDirectory = picker
        storage.session_dir = real_session_dir
        rehapose.settings().setValue("dataDir", tmp)


def check_settings(window):
    window.age.setValue(81)
    window.cue.setChecked(False)
    window.save_settings()
    assert rehapose.settings().value("age", type=int) == 81
    assert rehapose.settings().value("exercise", type=str) == "knee_flexion"
    assert rehapose.settings().value("beep", True, type=bool) is False


if __name__ == "__main__":
    main()
