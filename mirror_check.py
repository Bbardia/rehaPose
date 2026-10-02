"""Mirror-consistency check on a recorded clip: how much does the pipeline disagree
with itself?

    python mirror_check.py clip.mp4 [--tier heavy|lite|both]

Flipping an image swaps the person's anatomical left and right, so a perfect pipeline
reports left_knee on the original equal to right_knee on the flipped copy. The gap is
the pipeline's own inconsistency - a floor on its error, not the error. It cannot see a
bias that is the same on both sides, and angles here are raw, before the One Euro filter.

Each stream gets its own VIDEO-mode landmarker, because that is the mode the app runs in
and VIDEO mode tracks across frames. Record the clip the way the app is used: side-on,
whole body in frame, doing the exercise.
"""
import argparse

import cv2
import numpy as np

from analysis import JOINTS, flexion, visible
from rehapose import TIERS, ensure_model


def landmarker(tier):
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    return vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(ensure_model(tier))),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))


def angles(result):
    if not result.pose_world_landmarks:
        return {}
    world, pixel = result.pose_world_landmarks[0], result.pose_landmarks[0]
    return {j: flexion(world, j) for j in JOINTS if visible(pixel, j)}


def mirror_gaps(path, tier):
    """{joint family: [abs gap per frame]} for one tier over the whole clip."""
    import mediapipe as mp
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    straight, mirrored = landmarker(tier), landmarker(tier)
    gaps = {j.split("_", 1)[1]: [] for j in JOINTS}
    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            stamp = int(frames * 1000 / fps) + 1
            frames += 1
            pair = []
            for model, image in ((straight, frame), (mirrored, cv2.flip(frame, 1))):
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pair.append(angles(model.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), stamp)))
            a, b = pair
            for joint in JOINTS:
                side, family = joint.split("_", 1)
                twin = ("right_" if side == "left" else "left_") + family
                if a.get(joint) is not None and b.get(twin) is not None:
                    gaps[family].append(abs(a[joint] - b[twin]))
    finally:
        cap.release()
        straight.close()
        mirrored.close()
    return gaps, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clip")
    parser.add_argument("--tier", choices=(*TIERS, "both"), default="both")
    args = parser.parse_args()
    tiers = TIERS if args.tier == "both" else (args.tier,)
    print(f"{'tier':6} {'joint':9} {'pairs':>6} {'median':>8} {'p90':>8}   (degrees)")
    for tier in tiers:
        gaps, frames = mirror_gaps(args.clip, tier)
        for family, values in gaps.items():
            if values:
                med, p90 = np.percentile(values, [50, 90])
                print(f"{tier:6} {family:9} {len(values):6d} {med:8.1f} {p90:8.1f}")
            else:
                print(f"{tier:6} {family:9} {0:6d} {'-':>8} {'-':>8}")
        print(f"{tier:6} {frames} frames read")


if __name__ == "__main__":
    main()
