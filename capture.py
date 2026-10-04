"""Camera capture, pose inference and the pinned models, shared with offline tools."""
import base64
import contextlib
import hashlib
import importlib.metadata
import io
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PyQt5 import QtCore

from analysis import TierRatchet, from_coco17

# Model cache stays in ~/.cache: it is re-downloadable, unlike sessions.
MODEL_DIR = Path.home() / ".cache" / "rehapose"
# Pinned to version 1 + bucket MD5: a model change underneath would void repeatability runs.
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{t}/float16/1/pose_landmarker_{t}.task")
MODEL_MD5 = {"heavy": "RT3sTQLMxNPOgStt6E+lFg==", "lite": "BKdd33yBGsehpFIyZt19iA=="}
# Two tiers: the ratchet only asks whether this machine can sustain heavy.
TIERS = ("heavy", "lite")
# Optional RTMPose (2D) tiers, most accurate first; one small detector serves all of them.
RTM_TIERS = ("x", "m", "s")
RTM_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
RTM_MIRROR = "https://huggingface.co/Tau-J/RTMPose/resolve/main/rtmposev1/onnx_sdk/"
RTM_FILES = {  # name: (zip under RTM_URL, sha256 of its end2end.onnx, model input w x h)
    "det": ("yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
            "ceb11c07298f95c50d7c5abeb906d03340c85f23aa79e3e66966e7fb6c307250", (416, 416)),
    "x": ("rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip",
          "df0c0fa91e9870b1515dcaff741fd76cc753dcfb12786862f61b243bae81cd52", (288, 384)),
    "m": ("rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
          "5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c", (192, 256)),
    "s": ("rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",
          "9aeb635b83f86aea45cf45d85798f7eba1a162de8e0d721c44e54fe5eebaf47d", (192, 256)),
}
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


def sha256_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_rtm(name):
    """Path of a pinned RTMPose-family ONNX model, fetched and checked like the MediaPipe ones."""
    zip_name, sha, _size = RTM_FILES[name]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"rtm-{name}.onnx"
    if path.exists() and sha256_of(path) != sha:
        path.unlink()
    if not path.exists():
        tmp = path.with_suffix(".part")
        for base in (RTM_URL, RTM_MIRROR):  # the mirror only if OpenMMLab's server is down
            try:
                with urllib.request.urlopen(base + zip_name, timeout=30) as response:
                    archive = zipfile.ZipFile(io.BytesIO(response.read()))
                break
            except OSError:
                if base == RTM_MIRROR:
                    raise
        tmp.write_bytes(archive.read(next(n for n in archive.namelist()
                                          if n.endswith("end2end.onnx"))))
        if sha256_of(tmp) != sha:
            tmp.unlink()
            raise RuntimeError(f"The RTMPose {name} model download was corrupted - try again.")
        tmp.rename(path)
    return path


def rtm_device():
    """Where RTMPose runs: 'cuda' (NVIDIA), 'mps' (Apple CoreML) or 'cpu'; None if not installed."""
    try:
        import onnxruntime
        import rtmlib  # noqa: F401
    except ImportError:
        return None
    providers = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        return "cuda"
    return "mps" if "CoreMLExecutionProvider" in providers else "cpu"


class MediaPipeModel:
    """MediaPipe pose in VIDEO mode: 3D world landmarks in metres."""

    def __init__(self, path):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        self.mp, self.stamp = mp, 0
        self.landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(path)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1))

    def __call__(self, frame, stamp=None):
        """(pixel, world) for one BGR frame, each None without a pose."""
        self.stamp = self.stamp + 33 if stamp is None else stamp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.landmarker.detect_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), self.stamp)
        return (res.pose_landmarks[0] if res.pose_landmarks else None,
                res.pose_world_landmarks[0] if res.pose_world_landmarks else None)

    def close(self):
        self.landmarker.close()


class RTMPoseModel:
    """RTMPose 2D keypoints behind a small person detector, in MediaPipe's landmark layout."""

    def __init__(self, det_path, pose_path, pose_size, device):
        from rtmlib import YOLOX, RTMPose
        with contextlib.redirect_stdout(io.StringIO()):  # rtmlib prints every model it loads
            # CoreML cannot run YOLOX (a static-shape error), so only CUDA moves the detector.
            self.det = YOLOX(str(det_path), model_input_size=RTM_FILES["det"][2],
                             backend="onnxruntime", device="cuda" if device == "cuda" else "cpu")
            self.pose = RTMPose(str(pose_path), model_input_size=pose_size,
                                backend="onnxruntime", device=device)

    def __call__(self, frame, _stamp=None):
        """(pixel, world) for one BGR frame, each None without a person; no clock needed."""
        boxes = self.det(frame)
        if len(boxes) == 0:
            return None, None
        keypoints, scores = self.pose(frame, bboxes=boxes)
        best = int(np.argmax(scores.mean(axis=1)))  # the most confidently seen person
        return from_coco17(keypoints[best], scores[best], frame.shape[1], frame.shape[0])

    def close(self):
        pass


class PoseWorker(QtCore.QThread):
    """Camera capture + inference, off the GUI thread so the UI never stalls."""

    # bgr frame, (pixel, world) each None without a pose, inference ms
    ready = QtCore.pyqtSignal(object, object, float)
    status = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, camera=0, backend="mediapipe"):
        super().__init__()
        self.camera, self.backend = camera, backend
        self.tiers = RTM_TIERS if backend == "rtmpose" else TIERS
        self.device = rtm_device() if backend == "rtmpose" else "cpu"
        self.tier_log = [self.tiers[0]]  # every tier this session ran on
        self._stop = False

    def stop(self):
        self._stop = True

    def _make(self, files):
        return RTMPoseModel(*files, self.device) if self.backend == "rtmpose" else (
            MediaPipeModel(files))

    def _name(self, tier):
        if self.backend == "rtmpose":
            return f"RTMPose-{tier} 2D on {DEVICE_NAMES[self.device]}"
        return f"MediaPipe {tier}"

    def _open_camera(self):
        cap = cv2.VideoCapture(self.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            self.failed.emit(f"Could not open {self.camera}." if isinstance(self.camera, str)
                             else f"Could not open camera {self.camera}. Try --camera 1.")
            return None
        return cap

    def run(self):
        # Unhandled exceptions in QThread.run SIGABRT the process via PyQt5's qFatal (exit 134).
        try:
            self._run()
        except Exception as exc:                                  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    def _run(self):
        models = self._prepare_models()
        if models is None:
            return
        cap = self._open_camera()
        if cap is None:
            return

        model = self._make(models[self.tiers[0]])
        self.status.emit(f"Backend: {self._name(self.tiers[0])}")
        ratchet = TierRatchet(len(self.tiers), budget_ms=BUDGET_MS, window_s=RATCHET_S)
        misses = 0

        try:
            while not self._stop:
                ok, frame = cap.read()
                if not ok and isinstance(self.camera, str):
                    return  # end of a clip: finished -> Main.on_ended -> the normal Stop
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
                world, dt = self._infer(model, frame)

                tier = ratchet.update(dt, time.perf_counter(), world is not None)
                if tier is not None:
                    model.close()
                    model = self._make(models[self.tiers[tier]])
                    self.tier_log.append(self.tiers[tier])
                    self.status.emit(f"Backend: {self._name(self.tiers[tier])} "
                                     f"(auto: too slow for {self.tiers[tier - 1]})")
        finally:
            cap.release()
            with contextlib.suppress(Exception):
                model.close()

    def _prepare_models(self):
        """{tier: model files}, or None if Stop was pressed while fetching."""
        if self.backend == "rtmpose" and self.device is None:
            self.failed.emit("RTMPose is not installed - see README, 'More accurate model'.")
            return None
        # Fetch every tier before the camera so a step-down never stalls the frame loop.
        if self.backend == "rtmpose":
            self.status.emit("Preparing RTMPose models (the first run downloads ~275 MB)...")
        else:
            self.status.emit("Preparing models (the first run downloads about 36 MB)...")
        models = {}
        for tier in self.tiers:
            if self._stop:
                return None
            models[tier] = ((ensure_rtm("det"), ensure_rtm(tier), RTM_FILES[tier][2])
                            if self.backend == "rtmpose" else ensure_model(tier))
        return None if self._stop else models

    def _infer(self, model, frame):
        """One frame through the model and out to the UI; returns (world, inference ms)."""
        # Infer on the UNFLIPPED frame: mirroring first swaps every left_*/right_* label.
        t0 = time.perf_counter()
        pixel, world = model(frame)
        dt = (time.perf_counter() - t0) * 1000.0
        self.ready.emit(frame, (pixel, world), dt)
        return world, dt


DEVICE_NAMES = {"cuda": "NVIDIA GPU", "mps": "Apple GPU", "cpu": "CPU", None: "not installed"}


def mediapipe_version():
    try:
        return importlib.metadata.version("mediapipe")
    except importlib.metadata.PackageNotFoundError:
        return ""
