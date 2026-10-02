"""Camera capture, pose inference and the pinned models it runs on.

No widgets: PoseWorker is a QThread that emits frames and landmarks, and everything it
needs - which model, at which version, checked against which hash - lives here too, so
offline tools such as mirror_check.py get the exact same models without the UI.
"""
import base64
import contextlib
import hashlib
import importlib.metadata
import shutil
import time
import urllib.request
from pathlib import Path

import cv2
from PyQt5 import QtCore

from analysis import TierRatchet

# The model cache stays in ~/.cache - it is re-downloadable, which is what a cache is
# for. Sessions are the user's only copy and live wherever they chose on first run.
MODEL_DIR = Path.home() / ".cache" / "rehapose"
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
        # A timeout, because a stalled read with none blocks the worker - and Start, which
        # waits for a stopping worker - for as long as the network cares to take.
        with urllib.request.urlopen(url, timeout=30) as response, open(tmp, "wb") as out:
            shutil.copyfileobj(response, out)
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
        models = {}
        for tier in TIERS:
            if self._stop:          # Stop pressed mid-download: do not fetch the rest
                return
            models[tier] = ensure_model(tier)

        if self._stop:
            return
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
