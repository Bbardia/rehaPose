"""Joint angles, smoothing and metrics; no Qt/camera/mediapipe: `python analysis.py` self-checks."""
import math
import numpy as np

# MediaPipe Pose landmark indices, (proximal, vertex, distal).
JOINTS = {
    "left_knee":      (23, 25, 27),
    "right_knee":     (24, 26, 28),
    "left_hip":       (11, 23, 25),
    "right_hip":      (12, 24, 26),
    "left_elbow":     (11, 13, 15),
    "right_elbow":    (12, 14, 16),
    "left_shoulder":  (23, 11, 13),
    "right_shoulder": (24, 12, 14),
}

# Clinical convention: 0 deg = full extension (supplement of interior), except shoulders.
SUPPLEMENT = {j: not j.endswith("shoulder") for j in JOINTS}

# Hardcoded: mediapipe 1.0 has no mp.solutions.
CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 31), (28, 32),
]


def flexion(landmarks, joint):
    """Clinical flexion in degrees; feed pose_world_landmarks, never pose_landmarks."""
    ia, ib, ic = JOINTS[joint]
    a, b, c = (np.array([p.x, p.y, p.z]) for p in (landmarks[ia], landmarks[ib], landmarks[ic]))
    v1, v2 = a - b, c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-9:
        return None
    interior = math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))))
    return 180.0 - interior if SUPPLEMENT[joint] else interior


def visible(landmarks, joint, threshold=0.5):
    return all(landmarks[i].visibility >= threshold for i in JOINTS[joint])


# COCO-17 keypoint k (RTMPose and other 2D models) -> its MediaPipe landmark index.
COCO17_TO_MEDIAPIPE = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)


class Point:
    """A landmark, as the angle and setup code reads MediaPipe's own."""

    def __init__(self, x, y, z=0.0, visibility=0.0):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


class Planar(list):
    """World landmarks from a 2D model: pixels with z = 0, so angles lie in the image plane."""

    planar = True


def from_coco17(keypoints, scores, frame_w, frame_h, margin=0.02):
    """(pixel, world) in MediaPipe's layout; a point off or on the frame edge counts as hidden."""
    pixel = [Point(0.5, 0.5) for _ in range(33)]
    world = Planar(Point(0.0, 0.0) for _ in range(33))
    for k, i in enumerate(COCO17_TO_MEDIAPIPE):
        x, y = float(keypoints[k][0]), float(keypoints[k][1])
        inside = (margin * frame_w <= x <= (1 - margin) * frame_w
                  and margin * frame_h <= y <= (1 - margin) * frame_h)
        pixel[i] = Point(x / frame_w, y / frame_h, 0.0, float(scores[k]) if inside else 0.0)
        world[i] = Point(x, y, 0.0)  # pixels, never normalized: that would shear every angle
    return pixel, world


class OneEuro:
    """One Euro filter (Casiez et al. 2012): kills jitter at rest without smearing rep peaks."""

    def __init__(self, mincutoff=1.0, beta=0.02, dcutoff=1.0):
        self.mincutoff, self.beta, self.dcutoff = mincutoff, beta, dcutoff
        self.x_prev = self.dx_prev = self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t_prev is None or t <= self.t_prev:
            self.x_prev, self.dx_prev, self.t_prev = x, 0.0, t
            return x
        dt = t - self.t_prev
        dx = (x - self.x_prev) / dt
        dx_hat = self.dx_prev + self._alpha(self.dcutoff, dt) * (dx - self.dx_prev)
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        x_hat = self.x_prev + self._alpha(cutoff, dt) * (x - self.x_prev)
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class TierRatchet:
    """Picks the model tier by timing frames with a person; only steps down, early."""

    def __init__(self, n_tiers, budget_ms=40.0, window_s=5.0, samples=30):
        self.n_tiers, self.budget, self.window, self.samples = n_tiers, budget_ms, window_s, samples
        self.tier = 0
        self.locked = n_tiers < 2
        self.probe = []
        self.started = None

    def update(self, dt_ms, now, detected):
        """Feed one frame. Returns the new tier index if it changed, else None."""
        if self.locked or not detected:
            return None
        if self.started is None:
            self.started = now
        self.probe.append(dt_ms)
        if now - self.started >= self.window:
            self.locked = True
            return None
        if len(self.probe) < self.samples:
            return None
        if self._fast_enough():
            self.locked = True
            return None
        self.tier += 1
        self.probe.clear()
        self.started = now
        return self.tier

    def _fast_enough(self):
        """The recent median fits the budget, or there is no lighter tier left to try."""
        median = float(np.median(self.probe[-self.samples:]))
        return median <= self.budget or self.tier >= self.n_tiers - 1


def rep_spans(angles, lo=None, hi=None, lo_frac=0.3, hi_frac=0.7, min_range=15.0):
    """Hysteresis rep spans (trough, end) into the ORIGINAL list; lo/hi for protocol tests."""
    vals = [(i, v) for i, v in enumerate(angles) if v is not None]
    if len(vals) < 3:
        return []
    thresholds = _thresholds(vals, lo, hi, lo_frac, hi_frac, min_range)
    if thresholds is None:  # noise, not movement
        return []
    lo, hi = thresholds

    spans, armed = [], False
    start, trough = vals[0]
    for i, v in vals:
        if not armed:
            if v <= trough:
                trough, start = v, i
            if v >= hi:
                armed = True
        elif v <= lo:
            spans.append((start, i))
            armed = False
            trough, start = v, i
    return spans


def _thresholds(vals, lo, hi, lo_frac, hi_frac, min_range):
    """(lo, hi), filling whichever is None from the observed range; None if it is flat."""
    if lo is not None and hi is not None:
        return lo, hi
    # Percentiles, not min/max: one glitched frame would push hi above every rep.
    seen = [v for _, v in vals]
    lo_v, hi_v = (float(x) for x in np.percentile(seen, [5, 95]))
    if hi_v - lo_v < min_range:
        return None
    lo = lo_v + lo_frac * (hi_v - lo_v) if lo is None else lo
    hi = lo_v + hi_frac * (hi_v - lo_v) if hi is None else hi
    return lo, hi


def count_reps(angles, **kwargs):
    return len(rep_spans(angles, **kwargs))


def summarize(angles, lo=None, hi=None):
    """Per-joint session summary. `angles` may contain None for undetected frames."""
    seen = [v for v in angles if v is not None]
    if not seen:
        return {"rom": 0.0, "peak": 0.0, "min": 0.0, "reps": 0, "coverage": 0.0}
    a = np.asarray(seen, dtype=float)
    spans = rep_spans(angles, lo=lo, hi=hi)
    # Median of per-rep ranges so one glitched frame cannot inflate ROM.
    roms = _rep_roms(angles, spans)
    return {
        "rom": float(np.median(roms)) if roms else float(a.max() - a.min()),
        "peak": float(a.max()),
        "min": float(a.min()),
        "reps": len(spans),
        "coverage": 100.0 * len(seen) / len(angles),
    }


def _rep_roms(angles, spans):
    """Range of each span, over its detected frames only."""
    segs = ([v for v in angles[s:e + 1] if v is not None] for s, e in spans)
    return [max(seg) - min(seg) for seg in segs if seg]


# --- Setup check -----------------------------------------------------------------

def orientation_cos(pixel, world, frame_w, frame_h):
    """~1.0 facing the camera, ~0.0 side-on; MUST use pixel coords, not normalized."""
    if getattr(world, "planar", False):
        return None  # a 2D model has no depth, so no turn to measure
    torso_px = abs((pixel[11].y + pixel[12].y) / 2 - (pixel[23].y + pixel[24].y) / 2) * frame_h
    torso_m = abs((world[11].y + world[12].y) / 2 - (world[23].y + world[24].y) / 2)
    if torso_px < 1e-6 or torso_m < 1e-9:
        return None
    ratios = []
    # Average shoulders and hips: each 3D width leans on noisy z.
    for a, b in ((11, 12), (23, 24)):
        span_px = abs(pixel[a].x - pixel[b].x) * frame_w
        # True 3D width, not world x: world x shrinks with yaw and cancels the turn.
        span_m = math.dist((world[a].x, world[a].y, world[a].z),
                           (world[b].x, world[b].y, world[b].z))
        if span_m > 1e-9:
            ratios.append((span_px / torso_px) / (span_m / torso_m))
    return min(1.0, sum(ratios) / len(ratios)) if ratios else None


def in_view(pixel, joints, margin=0.02):
    """The joints whose three landmarks are confidently visible and clear of the frame edge."""
    return [j for j in joints if visible(pixel, j) and all(
        margin <= pixel[i].x <= 1 - margin and margin <= pixel[i].y <= 1 - margin
        for i in JOINTS[j])]


def setup_check(pixel, world, frame_w, frame_h, joints=tuple(JOINTS), view=None, margin=0.02):
    """(ok, hint) for the exercise's joints and camera angle; advice, never a lock."""
    if pixel is None or world is None:
        return False, "No person detected - step into frame."
    if not in_view(pixel, joints, margin):
        parts = sorted({j.split("_", 1)[1] + "s" for j in joints})
        names = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " or " + parts[-1]
        return False, f"Get your {names} fully in frame."
    hint = _angle_hint(pixel, world, frame_w, frame_h, view)
    return (False, hint) if hint else (True, "Setup looks good.")


def _angle_hint(pixel, world, frame_w, frame_h, view):
    """Advice if the camera sees the movement from the wrong side, else None (also if unsure)."""
    if not view or any(pixel[i].visibility < 0.5 for i in (11, 12, 23, 24)):
        return None
    cos = orientation_cos(pixel, world, frame_w, frame_h)
    if cos is None:
        return None
    if view == "side" and cos > 0.5:
        return f"Turn side-on to the camera (currently {round(cos * 100)}% front-on)."
    if view == "front" and cos < 0.8:
        return f"Turn to face the camera (currently {round(cos * 100)}% front-on)."
    return None


# (key, display, scored). KEYS ARE FROZEN (written to every file): append, never rename.
EXERCISES = (
    ("chair_stand_30s",    "30-second chair stand", True),
    ("knee_flexion",       "Knee flexion / extension", False),
    ("hip_abduction",      "Hip abduction", False),
    ("shoulder_abduction", "Shoulder abduction", False),
    ("shoulder_flexion",   "Shoulder flexion", False),
    ("heel_slides",        "Heel slides", False),
    ("other",              "Other / unlabelled", False),
)
EXERCISE_DISPLAY = {key: display for key, display, _ in EXERCISES}
# Joints each exercise needs in view, and the camera angle that sees its plane of movement.
EXERCISE_SETUP = {
    "chair_stand_30s":    (("left_knee", "right_knee"), "side"),
    "knee_flexion":       (("left_knee", "right_knee"), "side"),
    "hip_abduction":      (("left_hip", "right_hip"), "front"),
    "shoulder_abduction": (("left_shoulder", "right_shoulder"), "front"),
    "shoulder_flexion":   (("left_shoulder", "right_shoulder"), "side"),
    "heel_slides":        (("left_knee", "right_knee", "left_hip", "right_hip"), "side"),
    "other":              (tuple(JOINTS), None),
}


# --- 30-Second Chair Stand -------------------------------------------------------

CHAIR_STAND_SECONDS = 30.0
CHAIR_STAND_LO = 20.0   # knee flexion below this = standing
CHAIR_STAND_HI = 70.0   # knee flexion above this = seated
# Halfway up is ~60 deg (hip height ~ thigh*cos(knee)), not the 45 deg midpoint.
CHAIR_STAND_HALF = 60.0

# Rikli & Jones independence criterion (via SRALab): (age_low, age_high): (women, men).
CHAIR_STAND_NORMS = {
    (60, 64): (15, 17), (65, 69): (15, 16), (70, 74): (14, 15), (75, 79): (13, 14),
    (80, 84): (12, 13), (85, 89): (11, 11), (90, 94): (9, 9),
}


def chair_stand_score(knee, lo=CHAIR_STAND_LO, hi=CHAIR_STAND_HI):
    """(stands, final_counted); per protocol a final stand counts if past halfway up at 30 s."""
    spans = rep_spans(knee, lo=lo, hi=hi)
    tail = [v for v in knee[spans[-1][1] if spans else 0:] if v is not None]
    final = bool(tail) and max(tail) >= hi and tail[-1] <= CHAIR_STAND_HALF
    return len(spans) + final, final


def chair_stand_norm(age, sex):
    """Reference number of stands, or None if the age is unknown or outside the table."""
    if age is None or sex not in ("female", "male"):
        return None
    for (low, high), (women, men) in CHAIR_STAND_NORMS.items():
        if low <= age <= high:
            return women if sex == "female" else men
    return None


# --- Self-check ------------------------------------------------------------------

class _P:
    def __init__(self, x, y, z, v=1.0):
        self.x, self.y, self.z, self.visibility = x, y, z, v


def _cycles(n, a, b, steps):
    """n round trips a -> b -> a, `steps` samples each way."""
    out = []
    for _ in range(n):
        out += list(np.linspace(a, b, steps)) + list(np.linspace(b, a, steps))
    return out


def _body(yaw, w=1280, h=720, px_per_m=300.0):
    """(pixel, world) landmarks of a standing body turned by `yaw` degrees."""
    c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    joints = {0: (0, -0.75), 11: (0.2, -0.5), 12: (-0.2, -0.5), 23: (0.1, 0.0),
              24: (-0.1, 0.0), 25: (0.1, 0.45), 26: (-0.1, 0.45), 27: (0.1, 0.9),
              28: (-0.1, 0.9)}
    world, pixel = [_P(0, 0, 0)] * 33, [_P(0.5, 0.5, 0)] * 33
    for i, (x, y) in joints.items():
        world[i] = _P(x * c, y, -x * s_)          # yaw about the vertical axis
        pixel[i] = _P((w / 2 + px_per_m * x * c) / w, (h / 2 + px_per_m * y) / h, 0)
    return pixel, world


def _check_angles():
    lm = {23: _P(0, 0, 0), 25: _P(0, 1, 0), 27: _P(0, 2, 0)}
    lm = [lm.get(i, _P(0, 0, 0)) for i in range(33)]
    assert abs(flexion(lm, "left_knee") - 0.0) < 1e-6, flexion(lm, "left_knee")

    lm[27] = _P(1, 1, 0)
    assert abs(flexion(lm, "left_knee") - 90.0) < 1e-6, flexion(lm, "left_knee")

    # Shoulder is NOT supplemented.
    lm[23], lm[11], lm[13] = _P(0, 1, 0), _P(0, 0, 0), _P(0, 1, 0)
    assert abs(flexion(lm, "left_shoulder") - 0.0) < 1e-6, flexion(lm, "left_shoulder")

    lm[25].visibility = 0.1
    assert not visible(lm, "left_knee")


def _check_reps(wave):
    assert count_reps(wave) == 3, count_reps(wave)

    assert count_reps(list(45 + np.random.RandomState(0).randn(200) * 0.5)) == 0

    assert count_reps(list(np.linspace(0, 90, 50))) == 0


def _check_summary(wave):
    s = summarize(wave + [None] * 20)
    assert abs(s["rom"] - 90.0) < 1e-6 and s["reps"] == 3
    assert abs(s["coverage"] - 100 * 120 / 140) < 1e-6, s["coverage"]

    # ROM must survive a single glitched frame.
    glitched = list(wave)
    glitched[37] = 148.0
    assert abs(summarize(glitched)["rom"] - 90.0) < 1e-6, summarize(glitched)["rom"]
    assert max(glitched) - min(glitched) > 140.0


def _check_spans(wave):
    gappy = [None] * 5 + wave
    spans = rep_spans(gappy)
    assert len(spans) == 3 and spans[0][0] >= 5, spans
    assert all(gappy[s] is not None and gappy[e] is not None for s, e in spans)

    s0, e0 = rep_spans(wave)[0]
    seg = [v for v in wave[s0:e0 + 1] if v is not None]
    assert abs((max(seg) - min(seg)) - 90.0) < 1e-6


def _check_chair_stand():
    # Absolute thresholds: the self-calibrating default scores half-stands as full ones.
    full = _cycles(10, 90, 5, 15)
    half = _cycles(10, 90, 65, 15)
    assert count_reps(full, lo=CHAIR_STAND_LO, hi=CHAIR_STAND_HI) == 10
    assert count_reps(half, lo=CHAIR_STAND_LO, hi=CHAIR_STAND_HI) == 0
    assert count_reps(half) == 10

    three = full[:90]                                      # 3 stands, ends seated at 90
    assert chair_stand_score(three) == (3, False)
    assert chair_stand_score(three + list(np.linspace(90, 40, 8))) == (4, True)
    assert chair_stand_score(three + list(np.linspace(90, 55, 8))) == (4, True)
    assert chair_stand_score(three + list(np.linspace(90, 65, 8))) == (3, False)
    assert chair_stand_score(three + list(np.linspace(90, 5, 15))) == (4, False)
    assert chair_stand_score(list(np.linspace(90, 40, 8))) == (1, True)
    assert chair_stand_score([None, None]) == (0, False)


def _check_exercises_and_norms():
    # Keys are frozen: append, never edit.
    assert [k for k, _, _ in EXERCISES][:7] == [
        "chair_stand_30s", "knee_flexion", "hip_abduction", "shoulder_abduction",
        "shoulder_flexion", "heel_slides", "other"]
    assert len({k for k, _, _ in EXERCISES}) == len(EXERCISES), "duplicate exercise key"
    assert set(EXERCISE_SETUP) == {k for k, _, _ in EXERCISES}, "exercise without a setup"

    assert chair_stand_norm(72, "female") == 14
    assert chair_stand_norm(72, "male") == 15
    assert chair_stand_norm(30, "female") is None
    assert chair_stand_norm(None, "female") is None
    assert chair_stand_norm(72, None) is None       # unknown sex is not "male"


def _check_ratchet():
    r = TierRatchet(2, budget_ms=40.0)
    for i in range(60):
        assert r.update(10.0, i / 30.0, True) is None
    assert r.tier == 0 and r.locked

    r = TierRatchet(2, budget_ms=40.0)
    changed = [r.update(90.0, i / 30.0, True) for i in range(70)]
    assert 1 in changed, changed[:40]
    assert r.tier == 1

    # Undetected frames must not be timed.
    r = TierRatchet(2, budget_ms=40.0)
    for i in range(100):
        assert r.update(90.0, i / 30.0, False) is None
    assert r.tier == 0 and not r.locked and r.started is None

    r = TierRatchet(2, budget_ms=40.0, window_s=1.0)
    r.update(90.0, 0.0, True)
    assert r.update(90.0, 1.5, True) is None and r.locked


def _check_filter():
    f = OneEuro()
    for i in range(200):
        out = f(50.0, i / 30.0)
    assert abs(out - 50.0) < 1e-3, out
    f = OneEuro()
    outs = [f(v, i / 30.0) for i, v in enumerate(np.linspace(0, 90, 60))]
    assert max(outs) <= 90.0 + 1e-6 and outs[-1] > 80.0, (max(outs), outs[-1])


def _check_orientation():
    for yaw in (0, 30, 60, 85):
        pixel, world = _body(yaw)
        got = orientation_cos(pixel, world, 1280, 720)
        assert abs(got - abs(math.cos(math.radians(yaw)))) < 1e-6, (yaw, got)
    # Frame shape must not change the cosine.
    pixel_w, world_w = _body(60, 1280, 720)
    pixel_s, world_s = _body(60, 720, 720)
    assert abs(orientation_cos(pixel_w, world_w, 1280, 720)
               - orientation_cos(pixel_s, world_s, 720, 720)) < 1e-6


def _check_setup():
    knee, abduction = EXERCISE_SETUP["knee_flexion"], EXERCISE_SETUP["shoulder_abduction"]
    knees, shoulders = knee[0], abduction[0]
    assert setup_check(None, None, 1280, 720)[0] is False
    # Bending is filmed side-on, abduction facing the camera.
    assert setup_check(*_body(80), 1280, 720, *knee) == (True, "Setup looks good.")
    ok, hint = setup_check(*_body(0), 1280, 720, *knee)
    assert not ok and "side-on" in hint, hint
    assert setup_check(*_body(0), 1280, 720, *abduction)[0]
    ok, hint = setup_check(*_body(80), 1280, 720, *abduction)
    assert not ok and "face the camera" in hint, hint
    # Legs out of frame: fine for a shoulder exercise, not for a knee one.
    pixel, world = _body(0)
    for i in (25, 26, 27, 28):
        pixel[i] = _P(pixel[i].x, pixel[i].y, 0, v=0.1)
    assert setup_check(pixel, world, 1280, 720, shoulders, "front")[0]
    ok, hint = setup_check(pixel, world, 1280, 720, knees, "side")
    assert not ok and hint == "Get your knees fully in frame.", hint
    assert setup_check(pixel, world, 1280, 720)[0], "'other' needs any joint, not all"
    pixel, world = _body(80, px_per_m=420.0)               # feet run off the bottom
    assert in_view(pixel, shoulders) and not in_view(pixel, knees)
    pixel, world = _body(0)
    for i in (23, 24, 25, 26, 27, 28):
        pixel[i] = _P(pixel[i].x, pixel[i].y, 0, v=0.1)
    hint = setup_check(pixel, world, 1280, 720, EXERCISE_SETUP["heel_slides"][0], "side")[1]
    assert hint == "Get your hips or knees fully in frame.", hint


def _check_coco17():
    kp, sc = np.zeros((17, 2)), np.full(17, 0.9)
    kp[11], kp[13], kp[15] = (500, 300), (600, 400), (700, 300)  # left hip, knee, ankle
    pixel, world = from_coco17(kp, sc, 1280, 720)
    assert abs(flexion(world, "left_knee") - 90.0) < 1e-6, flexion(world, "left_knee")
    assert visible(pixel, "left_knee") and pixel[25].x == 600 / 1280
    assert orientation_cos(pixel, world, 1280, 720) is None
    assert setup_check(pixel, world, 1280, 720, ("left_knee",), "side")[0], "2D has no turn advice"
    assert pixel[31].visibility == 0, "COCO-17 has no foot index; it must never be drawn"
    kp[13] = (600, 715)  # a knee on the bottom edge is a guess, not a measurement
    pixel, _ = from_coco17(kp, sc, 1280, 720)
    assert not visible(pixel, "left_knee")


def demo():
    """Self-check: python analysis.py"""
    _check_angles()
    wave = _cycles(3, 0, 90, 20)
    _check_reps(wave)
    _check_summary(wave)
    _check_spans(wave)
    _check_chair_stand()
    _check_exercises_and_norms()
    _check_ratchet()
    _check_filter()
    _check_orientation()
    _check_setup()
    _check_coco17()
    print("analysis.py self-check passed")


if __name__ == "__main__":
    demo()
