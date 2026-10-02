"""Camera capture, pose inference and the pinned models, shared with offline tools."""
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

# Model cache stays in ~/.cache: it is re-downloadable, unlike sessions.
MODEL_DIR = Path.home() / ".cache" / "rehapose"
# Pinned to version 1 + bucket MD5: a model change underneath would void repeatability runs.
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{t}/float16/1/pose_landmarker_{t}.task")
MODEL_MD5 = {"heavy": "RT3sTQLMxNPOgStt6E+lFg==", "lite": "BKdd33yBGsehpFIyZt19iA=="}
# Two tiers: the ratchet only asks whether this machine can sustain heavy.
TIERS = ("heavy", "lite")
BUDGET_MS = 40.0    # 25 fps inference, room left for camera + plots
RATCHET_S = 5.0     # seconds of detected pose before the tier locks


def md5_of(path):
    return base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode()


def ensure_model(tier):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"pose_landmarker_{tier}.task"
    if path.exists() and md5_of(path) != MODEL_MD5[tier]:
        path.unlink()       # pre-pin or damaged cache
    if not path.exists():
        url = MODEL_URL.format(t=tier)
        tmp = path.with_suffix(".part")
        # Timeout: a stalled read would block the worker, and Start behind it, indefinitely.
        with urllib.request.urlopen(url, timeout=30) as response, open(tmp, "wb") as out:
            shutil.copyfileobj(response, out)
        if md5_of(tmp) != MODEL_MD5[tier]:
            tmp.unlink()
            raise RuntimeError(f"The {tier} model download was corrupted - try Start again.")
        tmp.rename(path)
    return path


class PoseWorker(QtCore.QThread):
    """Camera capture + inference, off the GUI thread so the UI never stalls."""

    # bgr frame, (pixel, world) each None without a pose, inference ms
    ready = QtCore.pyqtSignal(object, object, float)
    status = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, camera=0):
        super().__init__()
        self.camera = camera
        self.tier_log = [TIERS[0]]     # every tier this session ran on
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
        # Unhandled exceptions in QThread.run SIGABRT the process via PyQt5's qFatal (exit 134).
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

        models = self._prepare_models()
        if models is None:
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
                    # Denied macOS camera permission opens the device but never yields frames.
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
                stamp += 33
                world, dt = self._infer(mp, landmarker, frame, stamp)

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

    def _prepare_models(self):
        """{tier: model path}, or None if Stop was pressed while fetching."""
        # Fetch every tier before the camera so a step-down never stalls the frame loop.
        self.status.emit("Preparing models (the first run downloads about 36 MB)...")
        models = {}
        for tier in TIERS:
            if self._stop:
                return None
            models[tier] = ensure_model(tier)
        return None if self._stop else models

    def _infer(self, mp, landmarker, frame, stamp):
        """One frame through the model and out to the UI; returns (world, inference ms)."""
        # Infer on the UNFLIPPED frame: mirroring first swaps every left_*/right_* label.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        t0 = time.perf_counter()
        res = landmarker.detect_for_video(image, stamp)
        dt = (time.perf_counter() - t0) * 1000.0

        world = res.pose_world_landmarks[0] if res.pose_world_landmarks else None
        pixel = res.pose_landmarks[0] if res.pose_landmarks else None
        self.ready.emit(frame, (pixel, world), dt)
        return world, dt


def mediapipe_version():
    try:
        return importlib.metadata.version("mediapipe")
    except importlib.metadata.PackageNotFoundError:
        return ""
