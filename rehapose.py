"""rehaPose - live joint-angle biofeedback for rehab.

One window: camera with skeleton overlay on the left, eight live joint graphs on the
right, Start/Stop underneath, and a results table when you stop.

    python rehapose.py [--camera N]
"""
import argparse
import base64
import contextlib
import csv
import datetime
import hashlib
import importlib.metadata
import math
import os
import shutil
import struct
import sys
import time
import urllib.request
import wave
from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtMultimedia, QtWidgets, sip

from analysis import (CHAIR_STAND_HI, CHAIR_STAND_LO, CHAIR_STAND_SECONDS,
                      CONNECTIONS, EXERCISE_DISPLAY, EXERCISES, JOINTS, OneEuro,
                      TierRatchet, chair_stand_norm, chair_stand_score, count_reps,
                      flexion, setup_check, summarize, visible)

VERSION = "0.2"
# The model cache stays in ~/.cache - it is re-downloadable, which is what a cache is
# for. Sessions are the user's only copy and live wherever they chose on first run.
MODEL_DIR = Path.home() / ".cache" / "rehapose"
LEGACY_SESSIONS = MODEL_DIR / "sessions"
MIN_COVERAGE = 80.0   # below this a row is not reported as a measurement
AGE_UNSET = 17        # the age spinbox's "not entered" value, shown as "age ?"
TIPS = ("Stand side-on to the camera, with your whole body in frame.\n\n"
        "Good light in front of you, not behind. Clothing that shows your\n"
        "knees and hips reads better than loose trousers.\n\n"
        "Recording starts once the framing is right, not when you press Start.")


def settings():
    # No arguments: resolve from the QApplication's organization/application names.
    # Hardcoding ("rehaPose", "rehaPose") here made the smoke test's "rehaPoseTest"
    # name a no-op, so every test run pointed the REAL dataDir at a temp folder.
    return QtCore.QSettings()


def session_dir():
    """Where sessions live, or None if not chosen yet or no longer reachable."""
    chosen = settings().value("dataDir", "", type=str)
    if not chosen:
        return None
    path = Path(chosen) / "sessions"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None   # unplugged drive, revoked permission, or the folder became a file
    return path


def choose_data_dir(parent):
    """First run: ask once where sessions should live, then remember it.

    Asked rather than assumed because this is the user's only copy, and a folder they
    cannot find is a folder they cannot back up or send to anyone.
    """
    existing = session_dir()
    if existing is not None:
        return existing
    previous = settings().value("dataDir", "", type=str)
    default = Path(QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.DocumentsLocation)) / "rehaPose"
    QtWidgets.QMessageBox.information(
        parent, "rehaPose",
        (f"The sessions folder {previous} is not available - is a drive unplugged?\n\n"
         "Choose where to keep sessions. Pick the same folder again once it is back."
         if previous else
         "Choose a folder to keep your sessions in.\n\n"
         "Each session is a CSV you can open, back up or send on. Video is never "
         "saved and never leaves this machine."))
    picked = QtWidgets.QFileDialog.getExistingDirectory(
        parent, "Keep sessions in", str(default.parent))
    if not picked and previous:
        # Cancelled while the old folder is missing: keep pointing at it. Falling back to
        # Documents here would silently split sessions across two folders for good.
        return None
    root = Path(picked) if picked else default
    try:
        root.mkdir(parents=True, exist_ok=True)
        (root / "sessions").mkdir(exist_ok=True)
        moved = migrate_legacy(root / "sessions")
    except OSError as exc:
        QtWidgets.QMessageBox.warning(parent, "rehaPose",
                                      f"Cannot keep sessions in {root}: {exc.strerror or exc}")
        return None
    settings().setValue("dataDir", str(root))
    if moved:
        QtWidgets.QMessageBox.information(
            parent, "rehaPose", f"Moved {moved} earlier session(s) into {root}.")
    return session_dir()


def migrate_legacy(target):
    """Sessions used to be written under ~/.cache, which the OS may delete."""
    if not LEGACY_SESSIONS.is_dir():
        return 0
    target.mkdir(parents=True, exist_ok=True)
    moved = 0
    for old in LEGACY_SESSIONS.glob("*.csv"):
        new = target / old.name
        if new.exists():
            continue
        # Copy to a side name, then swap in atomically: rename() fails across volumes,
        # and shutil.move() can leave a truncated file under the real name that the
        # exists() check above would then skip forever.
        part = new.with_name(new.name + ".part")
        try:
            shutil.copy2(old, part)
            os.replace(part, new)
        finally:
            part.unlink(missing_ok=True)
        old.unlink()
        moved += 1
    return moved


# Version 1, not "latest": the tiers already differ by up to ~50 deg on one elbow, so a
# model that changes underneath a repeatability run makes the run meaningless. The MD5s
# are the bucket's own x-goog-hash for these exact objects (identical to "latest" as of
# 2026-10-02), so a truncated or swapped download is refused rather than trusted.
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{t}/float16/1/pose_landmarker_{t}.task")
MODEL_MD5 = {"heavy": "RT3sTQLMxNPOgStt6E+lFg==", "lite": "BKdd33yBGsehpFIyZt19iA=="}

# Two tiers, not three: the ratchet's only real question is "can this machine sustain
# heavy, yes or no". Measured on an M4: heavy 16.2 ms, lite 6.5 ms.
TIERS = ("heavy", "lite")
BUDGET_MS = 40.0    # 25 fps of inference, leaving room for camera + plots
RATCHET_S = 5.0     # after this many seconds of *detected pose*, the tier is locked
LIVE_WINDOW_S = 20.0
COUNTDOWN_S = 3     # before a scored test, so its 30 s starts on a signal, not on framing
# (Hz, ms). Distinct pitches so "go" and "time" are unmistakable without looking.
TONES = {"rep": (880, 90), "tick": (660, 120), "go": (1320, 350), "end": (440, 700)}


def md5_of(path):
    return base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode()


def ensure_model(tier):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"pose_landmarker_{tier}.task"
    if path.exists() and md5_of(path) != MODEL_MD5[tier]:
        path.unlink()       # cached before the pin, or damaged: fetch the pinned one
    if not path.exists():
        url = MODEL_URL.format(t=tier)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        if md5_of(tmp) != MODEL_MD5[tier]:
            tmp.unlink()
            raise RuntimeError(f"The {tier} model download was corrupted - try Start again.")
        tmp.rename(path)
    return path


class PoseWorker(QtCore.QThread):
    """Camera capture + inference, off the GUI thread so the UI never stalls."""

    # bgr frame, (pixel, world) - each None when no pose was found, inference ms
    ready = QtCore.pyqtSignal(object, object, float)
    status = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, camera=0):
        super().__init__()
        self.camera = camera
        self.tier_log = [TIERS[0]]     # every tier this session ran on, in order
        self._stop = False

    def stop(self):
        self._stop = True

    def _make(self, path):
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        return vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(path)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1))

    def _open_camera(self):
        cap = cv2.VideoCapture(self.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            self.failed.emit(f"Could not open camera {self.camera}. Try --camera 1.")
            return None
        return cap

    def run(self):
        # An unhandled exception in QThread.run does not raise into Qt - PyQt5 takes the
        # qFatal path and the whole process dies with SIGABRT. A first run with no
        # network reaches ensure_model() below, so without this the app simply
        # disappears. Reproduced: exit 134.
        try:
            self._run()
        except Exception as exc:                                  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    def _run(self):
        try:
            import mediapipe as mp
        except ImportError:
            self.failed.emit("mediapipe is not installed - see README")
            return

        # Every tier up front, before the camera: a ratchet step-down must not stall the
        # frame loop on a download while the measuring clock keeps running.
        self.status.emit("Preparing models (the first run downloads about 36 MB)...")
        models = {tier: ensure_model(tier) for tier in TIERS}

        cap = self._open_camera()
        if cap is None:
            return

        landmarker = self._make(models[TIERS[0]])
        self.status.emit(f"Backend: MediaPipe {TIERS[0]}")
        ratchet = TierRatchet(len(TIERS), budget_ms=BUDGET_MS, window_s=RATCHET_S)
        stamp, misses = 0, 0

        try:
            while not self._stop:
                ok, frame = cap.read()
                if not ok:
                    # A denied macOS camera permission opens the device but never
                    # yields a frame, so "keep retrying" would spin in silence forever.
                    misses += 1
                    if misses > 60:
                        self.failed.emit(
                            f"Camera {self.camera} opened but returned no frames. On macOS, "
                            "grant camera access to your terminal in System Settings > "
                            "Privacy & Security > Camera, then Start again.")
                        return
                    self.msleep(50)
                    continue
                misses = 0
                # Inference runs on the UNFLIPPED frame: MediaPipe infers anatomical
                # left/right from the image, so mirroring first silently swaps every
                # left_* and right_* label. Mirroring happens at display time instead.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                stamp += 33
                t0 = time.perf_counter()
                res = landmarker.detect_for_video(image, stamp)
                dt = (time.perf_counter() - t0) * 1000.0

                world = res.pose_world_landmarks[0] if res.pose_world_landmarks else None
                pixel = res.pose_landmarks[0] if res.pose_landmarks else None
                self.ready.emit(frame, (pixel, world), dt)

                tier = ratchet.update(dt, time.perf_counter(), world is not None)
                if tier is not None:
                    landmarker.close()
                    landmarker = self._make(models[TIERS[tier]])
                    self.tier_log.append(TIERS[tier])
                    self.status.emit(
                        f"Backend: MediaPipe {TIERS[tier]} (auto: too slow for "
                        f"{TIERS[tier - 1]})")
        finally:
            cap.release()
            with contextlib.suppress(Exception):
                landmarker.close()


def mediapipe_version():
    try:
        return importlib.metadata.version("mediapipe")
    except importlib.metadata.PackageNotFoundError:
        return ""


def make_beep(path, freq=880.0, ms=90, rate=44100):
    """Generate a cue as a .wav rather than shipping a binary asset. Cached by path, so
    the filename must encode the tone."""
    if path.exists():
        return path
    n = int(rate * ms / 1000)
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(b"".join(
            # fade the tail out, otherwise the abrupt cut clicks
            struct.pack("<h", int(22000 * math.sin(2 * math.pi * freq * i / rate)
                                  * min(1.0, 4.0 * (n - i) / n)))
            for i in range(n)))
    return path


def stamp_of(path):
    """Display date from the filename, which is where the timestamp already lives."""
    try:
        return datetime.datetime.strptime(path.name[:15], "%Y%m%d-%H%M%S").strftime(
            "%Y-%m-%d %H:%M")
    except ValueError:
        return path.stem


def is_blank(row):
    # A spreadsheet re-save pads blank rows with commas out to the used width.
    return not any(cell.strip() for cell in row)


def read_header(path):
    """Block 1 of a session file as a dict. Files are written with csv.writer, so they
    must be read back with csv.reader on newline="" - the line endings are CRLF."""
    head = {}
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as handle:
        for row in csv.reader(handle):
            if is_blank(row):
                break
            if len(row) >= 2:
                head[row[0]] = row[1]
            if len(row) >= 4:
                head[row[2]] = row[3]
    return head


def read_joints(path):
    """Block 2 of a session file: {joint: {column: value}}."""
    rows, section, header = {}, 0, None
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as handle:
        for row in csv.reader(handle):
            if is_blank(row):
                section += 1
                continue
            if section == 1:
                if row[0] == "joint":
                    header = row
                    continue
                if header is None:
                    raise ValueError("per-joint block has no header row")
                rows[row[0]] = dict(zip(header[1:], row[1:], strict=False))
    return rows


def big_text(frame, text):
    """Large outlined text in the top-left corner - drawn AFTER the mirror flip, or it
    reads backwards."""
    h = frame.shape[0]
    scale, org = h / 160, (int(h * 0.04), int(h * 0.22))
    for color, width in (((0, 0, 0), 18), ((255, 255, 255), 6)):
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, width,
                    cv2.LINE_AA)
    return frame


def stored_summaries(path):
    """Block 2 in the shape summarize() returns, so one renderer serves live and stored.
    Raises ValueError/KeyError on a file that has been edited out of shape."""
    return {joint: {"rom": float(v["rom_deg"]), "peak": float(v["peak_deg"]),
                    "min": float(v["min_deg"]), "reps": int(v["reps"]),
                    "coverage": float(v["tracked_pct"])}
            for joint, v in read_joints(path).items()}


def draw_overlay(frame, pixel, angles):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in pixel]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (245, 180, 60), 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (60, 220, 255), -1, cv2.LINE_AA)
    for joint, value in angles.items():
        if value is None:
            continue
        x, y = pts[JOINTS[joint][1]]
        cv2.putText(frame, f"{value:.0f}", (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


class Main(QtWidgets.QMainWindow):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.worker = None
        self.backend = ""
        self.t0 = 0.0
        self.times = {j: [] for j in JOINTS}
        self.angles = {j: [] for j in JOINTS}
        self.filters = {}
        self.stands = []        # chair-stand signal: max(left knee, right knee) per frame
        self.stand_count = 0
        self.clock_start = None  # measurement clock: starts when setup is good, not at Start
        self.setup_ok = 0
        self.frames = 0
        self.summaries = {}
        self.viewing_stored = None
        self.model_tier = ""
        self.unsaved = False     # the last session's autosave failed and no copy exists
        self.retiring = None     # a stopped worker that outlived its 2 s wait
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.sounds = {}
        for name, (freq, ms) in TONES.items():
            self.sounds[name] = QtMultimedia.QSoundEffect()
            self.sounds[name].setSource(QtCore.QUrl.fromLocalFile(str(
                make_beep(MODEL_DIR / f"tone-{freq}-{ms}.wav", freq, ms))))
        self.counted = None     # last countdown number shown, None outside a countdown
        self.time_called = self.final_counted = False
        self.setWindowTitle("rehaPose")
        self.resize(1500, 820)

        self.video = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video.setMinimumWidth(640)
        self.video.setStyleSheet("background:#111;color:#888")
        self.video.setText(TIPS)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self._build_plots())
        self.results = QtWidgets.QTableWidget(len(JOINTS), 6)
        self.results.setHorizontalHeaderLabels(
            ["Joint", "ROM", "Peak", "Min", "Reps", "Tracked"])
        self.results.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch)
        self.results.verticalHeader().setVisible(False)
        self.stack.addWidget(self.results)

        self.exercise = QtWidgets.QComboBox()
        for key, display, _ in EXERCISES:
            self.exercise.addItem(display, key)
        self.exercise.currentIndexChanged.connect(self.on_exercise_changed)
        self.person = QtWidgets.QLineEdit()
        self.person.setPlaceholderText("person code")
        self.person.setToolTip("Initials or a code, not a name - it is written into "
                               "every session file.")
        self.person.setMaxLength(24)
        self.person.setFixedWidth(130)
        self.person.editingFinished.connect(self.on_person_changed)
        self.last_person = ""
        self.age = QtWidgets.QSpinBox()
        self.age.setRange(AGE_UNSET, 99)
        self.age.setSpecialValueText("age ?")
        self.age.setValue(70)
        self.age.setPrefix("age ")
        self.sex = QtWidgets.QComboBox()
        self.sex.addItems(["female", "male"])
        self.cue = QtWidgets.QCheckBox("Beep on rep")
        self.cue.setChecked(True)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Exercise:"))
        top.addWidget(self.exercise)
        top.addWidget(self.person)
        top.addWidget(self.age)
        top.addWidget(self.sex)
        top.addWidget(self.cue)
        top.addStretch(1)

        self.button = QtWidgets.QPushButton("Start")
        self.button.setMinimumHeight(40)
        self.button.clicked.connect(self.toggle)
        self.export = QtWidgets.QPushButton("Save a copy...")
        self.export.clicked.connect(self.save_csv)
        self.status = QtWidgets.QLabel("Pick an exercise and press Start.")

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(self.button, 2)
        bar.addWidget(self.export, 1)
        bar.addWidget(self.status, 5)

        split = QtWidgets.QHBoxLayout()
        split.addWidget(self.video, 5)
        split.addWidget(self.stack, 6)

        live = QtWidgets.QVBoxLayout()
        live.addLayout(top)
        live.addLayout(split, 1)
        live.addLayout(bar)
        live_page = QtWidgets.QWidget()
        live_page.setLayout(live)

        self.pages = QtWidgets.QStackedWidget()
        self.pages.addWidget(live_page)
        self.pages.addWidget(self._build_history())
        self.setCentralWidget(self.pages)

        self._build_menus()
        self.restore_settings()
        self.on_exercise_changed()
        self._snapshot()
        self._sync()

    def _build_menus(self):
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        self.act_save = file_menu.addAction("Save a Copy...", self.save_csv,
                                            QtGui.QKeySequence.Save)
        file_menu.addAction("Show Sessions Folder", self.open_folder)
        file_menu.addSeparator()
        # self.close(), NOT app.quit(): quit must go through closeEvent, or the worker
        # is killed mid-frame and the session is never written. Quitting would become
        # the one path in the app that loses data.
        file_menu.addAction("Quit", self.close, QtGui.QKeySequence.Quit)

        session_menu = bar.addMenu("&Session")
        self.act_run = session_menu.addAction("Start", self.toggle,
                                              QtGui.QKeySequence("Ctrl+R"))

        view_menu = bar.addMenu("&View")
        view_menu.addAction("Live", lambda: self.pages.setCurrentIndex(0),
                            QtGui.QKeySequence("Ctrl+1"))
        # Ctrl+2, not Ctrl+H: Cmd+H is Hide Application on macOS.
        self.act_history = view_menu.addAction("History", self.show_history,
                                               QtGui.QKeySequence("Ctrl+2"))

        help_menu = bar.addMenu("&Help")
        help_menu.addAction("How to Record", self.show_tips,
                            QtGui.QKeySequence.HelpContents)
        help_menu.addAction("About rehaPose", self.show_about)

    def show_tips(self):
        QtWidgets.QMessageBox.information(self, "How to Record", TIPS)

    def show_about(self):
        QtWidgets.QMessageBox.about(
            self, "About rehaPose",
            f"<b>rehaPose {VERSION}</b><br><br>"
            "Measures and records joint angles from one camera.<br><br>"
            "Video is processed on this machine and is never saved or uploaded. "
            "Only joint angles are written to disk.<br><br>"
            "rehaPose measures and records exercise performance. It does not "
            "assess, screen or diagnose anything.")

    def open_folder(self):
        target = choose_data_dir(self)
        if target is None:
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(target)))

    def restore_settings(self):
        s = settings()
        geometry = s.value("geometry")
        if geometry is not None:          # restoreGeometry(None) raises TypeError
            self.restoreGeometry(geometry)
        key = s.value("exercise", "", type=str)
        index = self.exercise.findData(key)
        if index >= 0:
            self.exercise.setCurrentIndex(index)
        self.age.setValue(s.value("age", 70, type=int))
        self.sex.setCurrentText(s.value("sex", "female", type=str))
        self.person.setText(s.value("person", "", type=str))
        self.last_person = self.person.text().strip()
        self.cue.setChecked(s.value("beep", True, type=bool))

    def save_settings(self):
        s = settings()
        s.setValue("geometry", self.saveGeometry())
        s.setValue("exercise", self.exercise.currentData())
        s.setValue("age", self.age.value())
        s.setValue("sex", self.sex.currentText())
        s.setValue("person", self.person.text().strip())
        s.setValue("beep", self.cue.isChecked())

    def _sync(self):
        """The single place a widget is enabled or retitled.

        State is derived, not stored: `worker` means recording, `summaries` means there
        are results to save. Scattering setEnabled calls through start/stop/on_failed is
        what left the exercise combo disabled forever after a camera error.
        """
        recording = self.worker is not None
        reviewing = bool(self.summaries) and self.viewing_stored is None
        self.button.setText("Stop" if recording else "Start")
        self.act_run.setText("Stop" if recording else "Start")
        for widget in (self.exercise, self.person, self.age, self.sex):
            widget.setEnabled(not recording)
        self.export.setEnabled(reviewing)
        self.act_save.setEnabled(reviewing)
        self.act_history.setEnabled(not recording)
        label = self.exercise.currentText()
        self.setWindowTitle(f"rehaPose - {label}" if recording else "rehaPose")

    def _build_plots(self):
        pg.setConfigOptions(antialias=False)
        widget = pg.GraphicsLayoutWidget()
        self.curves, self.plots = {}, {}
        for i, joint in enumerate(JOINTS):
            if i and i % 2 == 0:
                widget.nextRow()
            plot = widget.addPlot(title=joint.replace("_", " "))
            plot.setYRange(0, 180, padding=0)
            plot.setMouseEnabled(False, False)
            plot.hideButtons()
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.setLabel("left", "flex", units="deg")
            self.plots[joint] = plot
            self.curves[joint] = plot.plot(pen=pg.mkPen("#3cc8ff", width=2))
        return widget

    def toggle(self):
        if self.worker:
            self.stop()
        else:
            self.start()

    def _build_history(self):
        self.caption = QtWidgets.QLabel()
        self.caption.setWordWrap(True)
        self.sessions = QtWidgets.QTableWidget(0, 6)
        self.sessions.setHorizontalHeaderLabels(
            ["Date", "Person", "Exercise", "Length", "Result", "Framing"])
        self.sessions.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch)
        self.sessions.verticalHeader().setVisible(False)
        self.sessions.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.sessions.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.sessions.itemDoubleClicked.connect(self.open_stored)
        back = QtWidgets.QPushButton("Back to Live")
        back.clicked.connect(lambda: self.pages.setCurrentIndex(0))
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.caption)
        layout.addWidget(self.sessions, 1)
        layout.addWidget(back)
        page = QtWidgets.QWidget()
        page.setLayout(layout)
        return page

    def show_history(self):
        target = choose_data_dir(self)
        if target is None:
            return
        if self.unsaved:
            # The folder is reachable again (that is how we got here): retry the save.
            with contextlib.suppress(OSError):
                self.status.setText(f"Saved {self.autosave().name} (on retry)")
                self.unsaved = False
        rows = sorted(target.glob("*.csv"), reverse=True)
        self.sessions.setRowCount(len(rows))
        self.history_files = rows
        for row, path in enumerate(rows):
            # One file mangled by a spreadsheet re-save must cost one row, not the page.
            try:
                head = read_header(path)
            except (OSError, csv.Error):
                head = {"exercise": "(unreadable)"}
            key = head.get("exercise", head.get("mode", ""))
            best = head.get("best_rom", "-")
            if head.get("best_joint"):
                best = f"{head['best_joint'].replace('_', ' ')} {best}"
            cells = [
                stamp_of(path),
                head.get("person") or "-",
                EXERCISE_DISPLAY.get(key, key or "-"),
                head.get("duration_s", "-"),
                head.get("stands", "") and f"{head['stands']} stands"
                + (" (stopped early)" if head.get("complete") == "no" else "") or best,
                head.get("framing_good_pct", "-") + "%"
                if head.get("framing_good_pct") else "-",
            ]
            for col, text in enumerate(cells):
                self.sessions.setItem(row, col, QtWidgets.QTableWidgetItem(str(text)))
        if rows:
            self.caption.setText(
                f"{len(rows)} session(s) in {target}. Double-click one to see its "
                "numbers. Compare only sessions of the same exercise recorded from the "
                "same camera position - treat differences under about 10° as noise.")
        else:
            self.caption.setText(
                "No sessions yet.\n\nRecord one from the Live screen and it will be "
                "saved here automatically. Sessions are CSV files you can open in any "
                "spreadsheet.")
        self.pages.setCurrentIndex(1)

    def open_stored(self, item):
        if self.unsaved:
            QtWidgets.QMessageBox.warning(
                self, "rehaPose", "The last session was not saved. Save a copy first "
                "(File > Save a Copy) - opening another session would hide it.")
            return
        path = self.history_files[item.row()]
        try:
            head, stored = read_header(path), stored_summaries(path)
        except (OSError, csv.Error, ValueError, KeyError) as exc:
            QtWidgets.QMessageBox.warning(self, "rehaPose", f"Cannot read {path.name}: {exc}")
            return
        if not stored:
            QtWidgets.QMessageBox.warning(self, "rehaPose",
                                          f"{path.name} has no per-joint numbers.")
            return
        self._fill_results(stored)
        # Viewing a stored session must not enable Save a Copy: that button writes the
        # LIVE session, which would land under the stored session's filename.
        self.viewing_stored = path
        self.stack.setCurrentIndex(1)
        self.pages.setCurrentIndex(0)
        line = f"Showing {path.name} (stored). Press Start for a new session."
        if head.get("stands"):
            reference = head.get("reference") or "no published reference for this age"
            if head.get("complete") == "no":
                reference = "none shown - the test was stopped before 30 s"
            line = (f"Chair stands: {head['stands']}.  Reference for an independent "
                    f"{head.get('sex', '?')} aged {head.get('age', '?')}: {reference}.  "
                    + line)
        self.status.setText(line)
        self._sync()

    def on_person_changed(self):
        """Age and sex belong to a person, not to the machine. Carried over silently,
        they put the last patient's norm beside the next patient's count."""
        code = self.person.text().strip()
        if code == self.last_person:
            return
        self.last_person = code
        known = (settings().value("people") or {}).get(code)
        if known:
            self.age.setValue(int(known[0]))
            self.sex.setCurrentText(str(known[1]))
        else:
            self.age.setValue(AGE_UNSET)        # ask, rather than inherit

    def remember_person(self):
        rec = self.recorded
        if rec["person"] and rec["age"] is not None:
            people = dict(settings().value("people") or {})
            people[rec["person"]] = [rec["age"], rec["sex"]]
            settings().setValue("people", people)

    def on_exercise_changed(self):
        self.age.setVisible(self.chair_mode)
        self.sex.setVisible(self.chair_mode)

    @property
    def exercise_key(self):
        return self.exercise.currentData()

    @property
    def chair_mode(self):
        return bool({k: s for k, _, s in EXERCISES}.get(self.exercise_key))

    @property
    def measured(self):
        return any(self.times.values())

    def _snapshot(self):
        """What is being recorded, frozen at Start. The widgets unlock again on Stop, so
        reading them at save time filed a copy under whatever the combo said by then."""
        age = self.age.value()
        self.recorded = {"exercise": self.exercise_key, "chair": self.chair_mode,
                         "person": self.person.text().strip(),
                         "age": None if age == AGE_UNSET else age,
                         "sex": self.sex.currentText()}

    def confirm_discard(self):
        """True if it is fine to drop the current session. Only asks when the autosave
        failed, because then the results on screen are the only copy there is."""
        if not self.unsaved:
            return True
        box = QtWidgets.QMessageBox
        answer = box.question(
            self, "rehaPose", "The last session was NOT saved.\n\nSave a copy first?",
            box.Save | box.Discard | box.Cancel, box.Save)
        if answer == box.Save:
            self.save_csv()
            return not self.unsaved
        if answer == box.Discard:
            self.unsaved = False
            return True
        return False

    def start(self):
        if not self.confirm_discard():
            return
        old = self.retiring
        if old is not None and not sip.isdeleted(old) and old.isRunning():
            # Two workers at once would both download into the same .part file.
            self.status.setText("Still stopping the previous session - try again shortly.")
            return
        if choose_data_dir(self) is None:
            return
        self.times = {j: [] for j in JOINTS}
        self.angles = {j: [] for j in JOINTS}
        self.filters = {j: OneEuro() for j in JOINTS}
        self.stands, self.stand_count = [], 0
        self.counted, self.time_called, self.final_counted = None, False, False
        self.clock_start, self.setup_ok, self.frames = None, 0, 0
        self.summaries, self.viewing_stored = {}, None
        self.on_person_changed()     # Cmd+R from inside the field skips editingFinished
        self._snapshot()
        self.remember_person()
        for joint, curve in self.curves.items():
            curve.setData([], [])
            self.plots[joint].setTitle(joint.replace("_", " "))
        self.stack.setCurrentIndex(0)
        self.pages.setCurrentIndex(0)
        self.t0 = time.perf_counter()
        self.worker = PoseWorker(self.camera)
        self.worker.ready.connect(self.on_frame)
        self.worker.status.connect(self.on_status)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()
        self._sync()

    def _stale(self):
        """True for a signal from a worker that has already been let go. One frame is
        nearly always still queued when Stop is pressed; it used to land after the
        autosave, overwrite the "Saved" line, and could even count a stand."""
        sender = self.sender()
        return sender is not None and sender is not self.worker

    def _detach(self):
        worker, self.worker = self.worker, None
        if worker is not None:
            worker.stop()
            if not worker.wait(2000):
                # Still inside a model download or load. Destroying a running QThread is
                # a qFatal abort - whether Python's refcount does it now or the window's
                # teardown does it at quit - so hand it to C++ with no parent: it is
                # freed when it finishes, or simply never, if the app exits first.
                sip.transferto(worker, None)
                worker.finished.connect(worker.deleteLater)
                self.retiring = worker
            # "heavy->lite" if the ratchet stepped down: the tiers disagree by up to
            # ~50 deg on one joint, so the file must not credit it all to one model.
            self.model_tier = "->".join(worker.tier_log)

    def stop(self):
        if not self.worker:
            return
        self._detach()
        if not self.measured:
            # Nothing was ever measured. Writing this would put an all-zeros row at the
            # top of History.
            self.video.setText(TIPS)
            self.status.setText(
                "Stopped before the test started - nothing recorded."
                if self.clock_start is not None else
                "Nothing recorded - the framing never came good. See Help > How to Record.")
            self._sync()
            return
        self.show_results()
        self._sync()

    def on_status(self, message):
        if self._stale():
            return
        self.backend = message
        self.status.setText(message)

    def on_failed(self, message):
        if self._stale():
            return
        self._detach()
        if self.measured:
            # A camera that drops out mid-set used to take the whole recording with it.
            self.show_results()
            message += "\n\nThe session up to that point was kept.\n" + self.status.text()
        else:
            self.status.setText(message)
        self._sync()
        QtWidgets.QMessageBox.critical(self, "rehaPose", message)

    def gate(self, pixel, world, frame):
        """(ok, hint). Starts the measurement clock the first time setup is good.

        The clock deliberately does not start at the button press: otherwise the 30 s
        expires while the patient is still walking into frame, and a free session's
        timeline begins before anything was measurable.
        """
        h, w = frame.shape[:2]
        ok, hint = setup_check(pixel, world, w, h)
        if ok and self.clock_start is None:
            lead = COUNTDOWN_S if self.recorded["chair"] else 0
            self.clock_start = time.perf_counter() + lead
            self.t0 = self.clock_start
        # Only frames from the recording itself count towards the framing figure. Frames
        # spent walking into shot are not a measurement, and counting them diluted the
        # percentage by however long the user took to get into position.
        if self.clock_start is not None and time.perf_counter() >= self.clock_start:
            self.frames += 1
            self.setup_ok += ok
        return ok, hint

    def measure(self, pixel, world, now):
        current = {}
        for joint in JOINTS:
            value = None
            if world is not None and visible(pixel, joint):
                raw = flexion(world, joint)
                if raw is not None:
                    value = self.filters[joint](raw, now)
            current[joint] = value
            self.times[joint].append(now)
            self.angles[joint].append(value)
        # One leg occluded should not lose the set, so the chair-stand signal is
        # whichever knee is more flexed.
        knees = [current[j] for j in ("left_knee", "right_knee") if current[j] is not None]
        self.stands.append(max(knees) if knees else None)
        return current

    def on_frame(self, frame, landmarks, dt):
        if self._stale():
            return
        pixel, world = landmarks
        ok, hint = self.gate(pixel, world, frame)
        if self.clock_start is None or time.perf_counter() < self.clock_start:
            self.show_waiting(cv2.flip(frame, 1), hint)
            return
        if self.counted:            # first measured frame after a countdown
            self.counted = 0
            self.sounds["go"].play()

        now = time.perf_counter() - self.t0
        current = self.measure(pixel, world, now)

        if pixel is not None:
            frame = draw_overlay(frame, pixel, current)
        # Mirror at display time only, so the overlay still lines up and the
        # left/right labels stay anatomically correct.
        shown = cv2.flip(frame, 1)
        if self.recorded["chair"]:
            big_text(shown, str(self.stand_count))
        self.show_frame(shown)
        self.update_plots(now, current)

        note = "" if ok else f"   |   {hint}"
        if self.recorded["chair"]:
            self.tick_chair_stand(now, note)
        else:
            self.status.setText(f"{self.backend}  |  {dt:.0f} ms/frame  "
                                f"({1000 / max(dt, 1e-6):.0f} fps){note}")

    def show_waiting(self, shown, hint):
        """Before the clock: setup hints, then a 3-2-1 with a tone per number."""
        if self.clock_start is None:
            self.status.setText(f"Setup: {hint}")
        else:
            n = math.ceil(self.clock_start - time.perf_counter())
            if n != self.counted:
                self.counted = n
                self.sounds["tick"].play()
            big_text(shown, str(n))
            self.status.setText(f"Get ready - starting in {n}")
        self.show_frame(shown)

    def tick_chair_stand(self, now, note):
        """Count stands and run the 30 s clock, stopping the session when it expires."""
        count = count_reps(self.stands, lo=CHAIR_STAND_LO, hi=CHAIR_STAND_HI)
        if count > self.stand_count:
            self.stand_count = count
            if self.cue.isChecked():
                self.sounds["rep"].play()
        left = CHAIR_STAND_SECONDS - now
        if left <= 0:
            # Time is called: the protocol's final-stand rule applies now, and only now.
            self.stand_count, self.final_counted = chair_stand_score(self.stands)
            self.time_called = True
            self.sounds["end"].play()
            self.stop()
            return
        self.status.setText(f"Chair stand: {self.stand_count}   |   {left:.0f}s left{note}")

    def update_plots(self, now, current):
        cutoff = now - LIVE_WINDOW_S
        for joint in JOINTS:
            start = np.searchsorted(self.times[joint], cutoff)
            # None -> nan so pyqtgraph breaks the line where tracking was lost
            ys = [np.nan if v is None else v for v in self.angles[joint][start:]]
            self.curves[joint].setData(self.times[joint][start:], ys, connect="finite")
            self.plots[joint].setXRange(max(0.0, cutoff), max(LIVE_WINDOW_S, now), padding=0)
            if current[joint] is not None:
                self.plots[joint].setTitle(
                    f"{joint.replace('_', ' ')}  {current[joint]:.0f}°")

    def show_frame(self, frame):
        h, w = frame.shape[:2]
        image = QtGui.QImage(frame.data, w, h, 3 * w, QtGui.QImage.Format_BGR888)
        self.video.setPixmap(QtGui.QPixmap.fromImage(image).scaled(
            self.video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def show_results(self):
        self.summaries = {j: summarize(self.angles[j]) for j in JOINTS}
        self._fill_results(self.summaries)
        self.stack.setCurrentIndex(1)
        try:
            saved = f"Saved {self.autosave().name}"
            self.unsaved = False
        except OSError as exc:
            self.unsaved = True
            # A full disk or unplugged drive used to raise straight out of this slot -
            # PyQt5's qFatal path, exit 134, and the session gone with the process.
            saved = (f"NOT SAVED ({exc.strerror or exc}) - use File > Save a Copy "
                     "before you start another session.")
        self.status.setText(f"{self.verdict()}  {saved}")

    def _fill_results(self, summaries):
        empty = {"rom": 0.0, "peak": 0.0, "min": 0.0, "reps": 0, "coverage": 0.0}
        for row, joint in enumerate(JOINTS):
            s = summaries.get(joint, empty)
            thin = s["coverage"] < MIN_COVERAGE
            # Below the coverage floor the numbers are not reported at all. A dash is
            # honest; a number computed from a third of the frames is not.
            cells = [joint.replace("_", " ")] + (
                ["--", "--", "--", "--"] if thin else
                [f"{s['rom']:.0f}°", f"{s['peak']:.0f}°", f"{s['min']:.0f}°", str(s["reps"])]
            ) + [f"{s['coverage']:.0f}%"]
            for col, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if thin:
                    item.setForeground(QtGui.QBrush(QtGui.QColor("#a06060")))
                self.results.setItem(row, col, item)

    def verdict(self):
        total = max((t[-1] for t in self.times.values() if t), default=0.0)
        framing = 100.0 * self.setup_ok / max(self.frames, 1)
        rec = self.recorded
        if rec["chair"]:
            norm = chair_stand_norm(rec["age"], rec["sex"])
            reference = (f"Reference for an independent {rec['sex']} aged "
                         f"{rec['age']}: {norm}." if norm else
                         "Age not entered, so no reference is shown." if rec["age"] is None
                         else "No published reference for this age.")
            if not self.time_called:
                # A count from a test stopped at 12 s is not comparable to a 30 s norm,
                # so the norm is not printed beside it.
                reference = (f"Stopped at {total:.1f}s - not a 30-second score, so no "
                             "reference is shown.")
            elif self.final_counted:
                reference = ("Includes a final stand more than halfway up at 30 s, "
                             "as the protocol counts it.  " + reference)
            return (f"Chair stands: {self.stand_count}.  {reference}  "
                    f"Protocol: 43-45 cm chair against a wall, arms crossed at the chest, "
                    f"full stand each rep - the app cannot check this.  "
                    f"Framing good in {framing:.0f}% of frames.")
        return (f"Session: {total:.0f}s.  Framing good in {framing:.0f}% of frames.  "
                f"Rows below {MIN_COVERAGE:.0f}% tracked are not reported.")

    def autosave(self):
        """Every session is written on stop. Pressing Start again used to discard the
        previous one silently, with the only copy behind a Save dialog nobody clicked."""
        target = session_dir()
        if target is None:
            raise OSError("the sessions folder is not available")
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        key = self.recorded["exercise"]
        path, n = target / f"{stamp}-{key}.csv", 2
        while path.exists():           # two stops in one second must not overwrite
            path, n = target / f"{stamp}-{key}-{n}.csv", n + 1
        self.write_csv(path)
        return path

    def write_csv(self, path):
        joints = list(JOINTS)
        reported = {j: s for j, s in self.summaries.items() if s["coverage"] >= MIN_COVERAGE}
        # Named, because the largest ROM of eight joints is often not the joint the
        # exercise is about - on a knee session it can be the elbow, the noisiest one.
        best_joint = max(reported, key=lambda j: reported[j]["rom"], default="")
        best = reported[best_joint]["rom"] if best_joint else 0.0
        duration = max((t[-1] for t in self.times.values() if t), default=0.0)
        rec = self.recorded
        # utf-8-sig: the BOM is what makes Excel show the degree sign instead of mojibake.
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            # The KEY, never the display text: History shows this, and renaming a
            # label must not orphan every session recorded before the rename.
            writer.writerow(["exercise", rec["exercise"], "person", rec["person"]])
            writer.writerow(["duration_s", f"{duration:.0f}"])
            # Provenance, so a later analysis can tell which model produced which angles.
            writer.writerow(["app_version", VERSION, "model", self.model_tier])
            writer.writerow(["mediapipe", mediapipe_version()])
            writer.writerow(["best_rom", f"{best:.0f}°", "best_joint", best_joint])
            if rec["chair"]:
                writer.writerow(["stands", self.stand_count,
                                 "complete", "yes" if self.time_called else "no"])
                writer.writerow(["age", "" if rec["age"] is None else rec["age"],
                                 "sex", rec["sex"]])
                writer.writerow(["reference", chair_stand_norm(rec["age"], rec["sex"])])
            framing = 100.0 * self.setup_ok / max(self.frames, 1)
            writer.writerow(["framing_good_pct", f"{framing:.1f}"])
            writer.writerow([])
            writer.writerow(["joint", "rom_deg", "peak_deg", "min_deg", "reps", "tracked_pct"])
            for joint in joints:
                s = self.summaries[joint]
                writer.writerow([joint, f"{s['rom']:.1f}", f"{s['peak']:.1f}",
                                 f"{s['min']:.1f}", s["reps"], f"{s['coverage']:.1f}"])
            writer.writerow([])
            writer.writerow(["time_s"] + joints)
            times = self.times[joints[0]]
            for i, t in enumerate(times):
                row = [f"{t:.3f}"]
                for joint in joints:
                    v = self.angles[joint][i]
                    row.append("" if v is None else f"{v:.2f}")
                writer.writerow(row)

    def save_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save a copy", "rehapose_session.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            self.write_csv(path)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "rehaPose",
                                          f"Could not save {path}: {exc.strerror or exc}")
            return
        self.unsaved = False
        self.status.setText(f"Saved {path}")

    def closeEvent(self, event):
        # Quitting mid-session used to write the CSV and vanish in the same tick, so the
        # user never saw the results or learned a file existed. Ask, then show them.
        if self.worker is not None:
            answer = QtWidgets.QMessageBox.question(
                self, "rehaPose", "A session is still recording.\n\n"
                "Stop and save it, or keep recording?",
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.Save)
            if answer == QtWidgets.QMessageBox.Cancel:
                event.ignore()
                return
            self.stop()
            QtWidgets.QMessageBox.information(self, "rehaPose", self.status.text())
        # A failed autosave leaves the screen as the only copy; quitting would lose it.
        if not self.confirm_discard():
            event.ignore()
            return
        self.save_settings()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="rehaPose live joint-angle biofeedback")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    app.setOrganizationName("rehaPose")
    app.setApplicationName("rehaPose")
    app.setApplicationVersion(VERSION)
    window = Main(args.camera)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
