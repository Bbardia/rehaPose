"""Mirror-gap error floor: python mirror_check.py clip.mp4 [--backend rtmpose] [--tier x]"""
import argparse

import cv2
import numpy as np

from analysis import JOINTS, flexion, visible
from capture import (RTM_FILES, RTM_TIERS, TIERS, MediaPipeModel, RTMPoseModel, ensure_model,
                     ensure_rtm, rtm_device)


def make_model(backend, tier):
    """The app's own model adapter, so this measures exactly what the app runs."""
    if backend == "rtmpose":
        return RTMPoseModel(ensure_rtm("det"), ensure_rtm(tier), RTM_FILES[tier][2],
                            rtm_device() or "cpu")
    return MediaPipeModel(ensure_model(tier))


def angles(pixel, world):
    if world is None:
        return {}
    return {j: flexion(world, j) for j in JOINTS if visible(pixel, j)}


def mirrored_pair(straight, mirrored, frame, stamp):
    """Angles from one frame and from its mirror image, each through its own model."""
    return [angles(*model(image, stamp))
            for model, image in ((straight, frame), (mirrored, cv2.flip(frame, 1)))]


def pair_gaps(a, b):
    """(family, abs gap) for each joint seen in `a` whose twin is seen in mirrored `b`."""
    for joint in JOINTS:
        side, family = joint.split("_", 1)
        twin = ("right_" if side == "left" else "left_") + family
        if a.get(joint) is not None and b.get(twin) is not None:
            yield family, abs(a[joint] - b[twin])


def mirror_gaps(path, backend, tier):
    """{joint family: [abs gap per frame]} for one model tier over the whole clip."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Bogus fps (0, NaN, 90000) would repeat ms stamps, which VIDEO mode rejects.
    fps = fps if 0 < fps <= 1000 else 30.0
    straight, mirrored = make_model(backend, tier), make_model(backend, tier)
    gaps = {j.split("_", 1)[1]: [] for j in JOINTS}
    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            stamp = int(frames * 1000 / fps) + 1
            frames += 1
            a, b = mirrored_pair(straight, mirrored, frame, stamp)
            for family, gap in pair_gaps(a, b):
                gaps[family].append(gap)
    finally:
        cap.release()
        straight.close()
        mirrored.close()
    return gaps, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clip")
    parser.add_argument("--backend", choices=("mediapipe", "rtmpose"), default="mediapipe")
    parser.add_argument("--tier", choices=(*TIERS, *RTM_TIERS, "all"), default="all")
    args = parser.parse_args()
    own = RTM_TIERS if args.backend == "rtmpose" else TIERS
    if args.tier not in (*own, "all"):
        parser.error(f"--tier {args.tier} is not a {args.backend} tier: {', '.join(own)}")
    tiers = own if args.tier == "all" else (args.tier,)
    print(f"{'tier':6} {'joint':9} {'pairs':>6} {'median':>8} {'p90':>8}   (degrees)")
    for tier in tiers:
        gaps, frames = mirror_gaps(args.clip, args.backend, tier)
        for family, values in gaps.items():
            if values:
                med, p90 = np.percentile(values, [50, 90])
                print(f"{tier:6} {family:9} {len(values):6d} {med:8.1f} {p90:8.1f}")
            else:
                print(f"{tier:6} {family:9} {0:6d} {'-':>8} {'-':>8}")
        print(f"{tier:6} {frames} frames read")


if __name__ == "__main__":
    main()
