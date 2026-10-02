"""Joint angles, smoothing and post-session metrics.

Pure numpy + stdlib: no Qt, no mediapipe, no camera. That is deliberate — it keeps
`python analysis.py` runnable as a self-check on a box with no webcam and no display.
"""
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

# Clinical convention: 0 deg = full extension for knee/elbow/hip, so flexion is the
# supplement of the interior angle. Shoulder is already 0 deg with the arm at the side.
SUPPLEMENT = {j: not j.endswith("shoulder") for j in JOINTS}

# Drawn skeleton (torso + limbs). mediapipe.solutions was removed in 1.0, so the
# connection list lives here now.
CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 31), (28, 32),
]


def flexion(landmarks, joint):
    """Clinical flexion angle in degrees, from metric 3D world landmarks.

    Must be fed pose_world_landmarks, not pose_landmarks: normalized coords divide x
    by frame width and y by height independently, which shears every angle.
    """
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


class OneEuro:
    """One Euro filter (Casiez et al. 2012).

    Beats the old EWMA on the thing that matters here: it widens its own cutoff when the
    limb is moving, so it kills jitter at rest without smearing the peak of a rep — and
    a smeared peak is a wrong ROM number.
    """

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
    """Picks the pose-model tier by timing real inference, not by inspecting hardware.

    Three rules, each there for a reason:
      - only frames that actually found a person are timed, because MediaPipe
        short-circuits on an empty frame and reads ~2.6x optimistic;
      - the clock starts at the first detected pose, not at Start, so a user who walks
        into frame still gets measured;
      - it only steps down, and only inside the opening window, because switching models
        mid-set shifts the measured angles and the rep counter would read that step as
        real movement.
    """

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
        median = float(np.median(self.probe[-self.samples:]))
        if median <= self.budget or self.tier >= self.n_tiers - 1:
            self.locked = True
            return None
        self.tier += 1
        self.probe.clear()
        self.started = now
        return self.tier


def rep_spans(angles, lo=None, hi=None, lo_frac=0.3, hi_frac=0.7, min_range=15.0):
    """Index spans (start, end) of completed reps, into the ORIGINAL list.

    Hysteresis: arm above `hi`, complete below `lo`. Two thresholds, not one, so jitter
    around a single line cannot inflate the count. `start` tracks the true trough since
    the last rep rather than the last sample below `lo`, so a span covers the whole
    movement and its range is the real rep ROM.

    Thresholds default to the joint's own observed range (self-calibrating: a knee bend
    and a shoulder raise share nothing but going out and coming back). Pass absolute
    lo/hi for a protocol test, where scoring a half-rep as a rep is the failure mode.

    Indices refer to the ORIGINAL list including Nones, so they map back to frame times.
    """
    vals = [(i, v) for i, v in enumerate(angles) if v is not None]
    if len(vals) < 3:
        return []
    if lo is None or hi is None:
        # 5th/95th percentile, not min/max: a single glitched frame would otherwise drag
        # the arm-threshold above anything the movement actually reaches, and rep
        # detection collapses to zero rather than merely reporting a wrong range.
        seen = [v for _, v in vals]
        lo_v, hi_v = (float(x) for x in np.percentile(seen, [5, 95]))
        if hi_v - lo_v < min_range:  # noise, not movement
            return []
        lo = lo_v + lo_frac * (hi_v - lo_v) if lo is None else lo
        hi = lo_v + hi_frac * (hi_v - lo_v) if hi is None else hi

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


def count_reps(angles, **kwargs):
    return len(rep_spans(angles, **kwargs))


def summarize(angles, lo=None, hi=None):
    """Per-joint session summary. `angles` may contain None for undetected frames."""
    seen = [v for v in angles if v is not None]
    if not seen:
        return {"rom": 0.0, "peak": 0.0, "min": 0.0, "reps": 0, "coverage": 0.0}
    a = np.asarray(seen, dtype=float)
    spans = rep_spans(angles, lo=lo, hi=hi)
    # ROM is the MEDIAN of the per-rep ranges, not the session max-minus-min: a single
    # glitched frame would otherwise permanently inflate the headline number a
    # clinician reads. With no completed rep there is nothing to take a median of, so
    # fall back to the session extent.
    roms = []
    for s, e in spans:
        seg = [v for v in angles[s:e + 1] if v is not None]
        if seg:
            roms.append(max(seg) - min(seg))
    return {
        "rom": float(np.median(roms)) if roms else float(a.max() - a.min()),
        "peak": float(a.max()),
        "min": float(a.min()),
        "reps": len(spans),
        "coverage": 100.0 * len(seen) / len(angles),
    }


# --- Setup check -----------------------------------------------------------------
# Camera placement is the largest controllable error source in single-camera pose, and
# it is worth catching BEFORE a session rather than reporting as a coverage number
# afterwards, on a session already wasted.

def orientation_cos(pixel, world, frame_w, frame_h):
    """How square-on the shoulders are: ~1.0 facing the camera, ~0.0 perfectly side-on.

    Compares the shoulder-span:torso-height ratio as seen by the camera against the
    same ratio in metric 3D, which supplies this individual's own anatomy - so it needs
    no per-user calibration step.

    The image ratio MUST be built from pixel coordinates. Normalized x and y are divided
    by frame width and height separately, so using them raw makes this measure the
    frame's aspect ratio rather than the body: verified 0.387 to 1.247 on one unchanged
    pose, which is also impossible for a cosine.
    """
    span_px = abs(pixel[11].x - pixel[12].x) * frame_w
    torso_px = abs((pixel[11].y + pixel[12].y) / 2 - (pixel[23].y + pixel[24].y) / 2) * frame_h
    # The TRUE 3D shoulder width, not its x component. World landmarks are camera-
    # aligned, so world x shrinks with yaw exactly as the image does and the ratio
    # cancelled the very turn it exists to measure: ~1.0 at every yaw.
    span_m = math.dist((world[11].x, world[11].y, world[11].z),
                       (world[12].x, world[12].y, world[12].z))
    torso_m = abs((world[11].y + world[12].y) / 2 - (world[23].y + world[24].y) / 2)
    if torso_px < 1e-6 or span_m < 1e-9 or torso_m < 1e-9:
        return None
    return min(1.0, (span_px / torso_px) / (span_m / torso_m))


def setup_check(pixel, world, frame_w, frame_h, view="side", margin=0.02):
    """(ok, message) - is the camera placed well enough to start?

    A warning, not a lock: a clinician may deliberately shoot frontal for a valgus view.
    """
    if pixel is None or world is None:
        return False, "No person detected - step into frame."
    needed = [0, 11, 12, 23, 24, 25, 26, 27, 28]
    if any(pixel[i].visibility < 0.5 for i in needed):
        return False, "Whole body not visible - step back so head and feet are in frame."
    xs = [pixel[i].x for i in needed]
    ys = [pixel[i].y for i in needed]
    if min(xs) < margin or max(xs) > 1 - margin or min(ys) < margin or max(ys) > 1 - margin:
        return False, "Body is touching the edge of frame - step back or re-aim."
    cos = orientation_cos(pixel, world, frame_w, frame_h)
    if cos is None:
        return False, "Cannot read your orientation - step back into full view."
    if view == "side" and cos > 0.5:
        return False, f"Turn side-on to the camera (currently {round(cos * 100)}% front-on)."
    if view == "front" and cos < 0.8:
        return False, f"Turn to face the camera (currently {round(cos * 100)}% front-on)."
    return True, "Setup looks good."


# --- Exercise vocabulary ---------------------------------------------------------
# (key, display, scored). The KEY is written into every session file and is what
# History groups by, so THE KEYS ARE FROZEN FOREVER - renaming one orphans every
# session already recorded under it, with no migration short of rewriting every file
# on disk. Display text can be reworded freely. Add to the end; never reuse a key.
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


# --- 30-Second Chair Stand -------------------------------------------------------
# Scored in COUNTS, which is why it survives on a single camera: the movement is a large
# sagittal knee excursion, the best-conditioned thing MediaPipe measures, and the score
# only needs to know that a threshold was crossed, not where the joint was.

CHAIR_STAND_SECONDS = 30.0
CHAIR_STAND_LO = 20.0   # knee flexion below this = standing
CHAIR_STAND_HI = 70.0   # knee flexion above this = seated

# Rikli & Jones criterion for maintaining physical independence, via SRALab.
# (age_low, age_high): (women, men)
CHAIR_STAND_NORMS = {
    (60, 64): (15, 17), (65, 69): (15, 16), (70, 74): (14, 15), (75, 79): (13, 14),
    (80, 84): (12, 13), (85, 89): (11, 11), (90, 94): (9, 9),
}


def chair_stand_score(knee, lo=CHAIR_STAND_LO, hi=CHAIR_STAND_HI):
    """(stands, final_counted) when time is called.

    The published protocol counts a final stand if the participant is more than halfway
    up at 30 s. Without that rule the count sits one below the protocol the norms were
    built on, for anyone caught mid-rise. "Halfway" is the knee-angle midpoint of the two
    thresholds - ponytail: knee angle is not linear in seat height, but at 45 deg between
    a 70 deg seat and a 20 deg stand the rise is well past its halfway point either way.
    """
    spans = rep_spans(knee, lo=lo, hi=hi)
    tail = [v for v in knee[spans[-1][1] if spans else 0:] if v is not None]
    final = bool(tail) and max(tail) >= hi and tail[-1] <= (lo + hi) / 2
    return len(spans) + final, final


def chair_stand_norm(age, sex):
    """Reference number of stands, or None if the age is outside the published table."""
    for (low, high), (women, men) in CHAIR_STAND_NORMS.items():
        if low <= age <= high:
            return women if sex == "female" else men
    return None


def demo():
    """Self-check: python analysis.py"""

    class P:
        def __init__(self, x, y, z, v=1.0):
            self.x, self.y, self.z, self.visibility = x, y, z, v

    # Straight leg: hip (0,0,0) - knee (0,1,0) - ankle (0,2,0) => 0 deg flexion.
    lm = {23: P(0, 0, 0), 25: P(0, 1, 0), 27: P(0, 2, 0)}
    lm = [lm.get(i, P(0, 0, 0)) for i in range(33)]
    assert abs(flexion(lm, "left_knee") - 0.0) < 1e-6, flexion(lm, "left_knee")

    # Right angle at the knee => 90 deg flexion.
    lm[27] = P(1, 1, 0)
    assert abs(flexion(lm, "left_knee") - 90.0) < 1e-6, flexion(lm, "left_knee")

    # Shoulder is NOT supplemented: arm at side => ~0 deg.
    lm[23], lm[11], lm[13] = P(0, 1, 0), P(0, 0, 0), P(0, 1, 0)
    assert abs(flexion(lm, "left_shoulder") - 0.0) < 1e-6, flexion(lm, "left_shoulder")

    # visibility gate
    lm[25].visibility = 0.1
    assert not visible(lm, "left_knee")

    # Three clean reps of a 0->90 bend.
    wave = []
    for _ in range(3):
        wave += list(np.linspace(0, 90, 20)) + list(np.linspace(90, 0, 20))
    assert count_reps(wave) == 3, count_reps(wave)

    # Noise around a fixed angle is not a rep.
    assert count_reps(list(45 + np.random.RandomState(0).randn(200) * 0.5)) == 0

    # A signal that goes up but never comes back is not a completed rep.
    assert count_reps(list(np.linspace(0, 90, 50))) == 0

    s = summarize(wave + [None] * 20)
    assert abs(s["rom"] - 90.0) < 1e-6 and s["reps"] == 3
    assert abs(s["coverage"] - 100 * 120 / 140) < 1e-6, s["coverage"]

    # ROM must survive a single glitched frame. Session max-minus-min would report the
    # glitch; median-of-reps reports the movement.
    glitched = list(wave)
    glitched[37] = 148.0
    assert abs(summarize(glitched)["rom"] - 90.0) < 1e-6, summarize(glitched)["rom"]
    assert max(glitched) - min(glitched) > 140.0   # the naive number really is that bad

    # Spans index into the ORIGINAL list, gaps included, so they map back to frame times.
    gappy = [None] * 5 + wave
    spans = rep_spans(gappy)
    assert len(spans) == 3 and spans[0][0] >= 5, spans
    assert all(gappy[s] is not None and gappy[e] is not None for s, e in spans)

    # A span must cover the whole movement, so its range is the real rep ROM.
    s0, e0 = rep_spans(wave)[0]
    seg = [v for v in wave[s0:e0 + 1] if v is not None]
    assert abs((max(seg) - min(seg)) - 90.0) < 1e-6

    # Absolute thresholds: full stands count, half stands do not. This is the whole
    # reason the chair-stand test cannot use the self-calibrating default -- a frail
    # patient doing ten 25-degree half-stands would otherwise score a perfect ten.
    full = []
    half = []
    for _ in range(10):
        full += list(np.linspace(90, 5, 15)) + list(np.linspace(5, 90, 15))
        half += list(np.linspace(90, 65, 15)) + list(np.linspace(65, 90, 15))
    assert count_reps(full, lo=CHAIR_STAND_LO, hi=CHAIR_STAND_HI) == 10
    assert count_reps(half, lo=CHAIR_STAND_LO, hi=CHAIR_STAND_HI) == 0
    # ...and the self-calibrating default is exactly what gets it wrong:
    assert count_reps(half) == 10

    # Final-stand rule: more than halfway up at 30 s counts, less does not, and a
    # participant already standing when time is called gets nothing extra.
    three = full[:90]                                      # 3 stands, ends seated at 90
    assert chair_stand_score(three) == (3, False)
    assert chair_stand_score(three + list(np.linspace(90, 40, 8))) == (4, True)
    assert chair_stand_score(three + list(np.linspace(90, 55, 8))) == (3, False)
    assert chair_stand_score(three + list(np.linspace(90, 5, 15))) == (4, False)
    assert chair_stand_score(list(np.linspace(90, 40, 8))) == (1, True)  # first rise
    assert chair_stand_score([None, None]) == (0, False)

    assert chair_stand_norm(72, "female") == 14
    assert chair_stand_norm(72, "male") == 15
    assert chair_stand_norm(30, "female") is None

    # Ratchet: a fast machine locks on tier 0 and never steps down.
    r = TierRatchet(2, budget_ms=40.0)
    for i in range(60):
        assert r.update(10.0, i / 30.0, True) is None
    assert r.tier == 0 and r.locked

    # A slow machine steps down once, then locks at the last tier.
    r = TierRatchet(2, budget_ms=40.0)
    changed = [r.update(90.0, i / 30.0, True) for i in range(70)]
    assert 1 in changed, changed[:40]
    assert r.tier == 1

    # Undetected frames must not be timed, or an empty room picks the wrong tier.
    r = TierRatchet(2, budget_ms=40.0)
    for i in range(100):
        assert r.update(90.0, i / 30.0, False) is None
    assert r.tier == 0 and not r.locked and r.started is None

    # The window closes on wall-clock time even if frames keep arriving.
    r = TierRatchet(2, budget_ms=40.0, window_s=1.0)
    r.update(90.0, 0.0, True)
    assert r.update(90.0, 1.5, True) is None and r.locked

    # One Euro must converge to a constant and not overshoot a ramp.
    f = OneEuro()
    for i in range(200):
        out = f(50.0, i / 30.0)
    assert abs(out - 50.0) < 1e-3, out
    f = OneEuro()
    outs = [f(v, i / 30.0) for i, v in enumerate(np.linspace(0, 90, 60))]
    assert max(outs) <= 90.0 + 1e-6 and outs[-1] > 80.0, (max(outs), outs[-1])

    # Orientation: a body turned by `yaw` must read |cos(yaw)|, whatever the frame shape.
    def body(yaw, w=1280, h=720, px_per_m=300.0):
        c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        joints = {0: (0, -0.75), 11: (0.2, -0.5), 12: (-0.2, -0.5), 23: (0.1, 0.0),
                  24: (-0.1, 0.0), 25: (0.1, 0.45), 26: (-0.1, 0.45), 27: (0.1, 0.9),
                  28: (-0.1, 0.9)}
        world, pixel = [P(0, 0, 0)] * 33, [P(0.5, 0.5, 0)] * 33
        for i, (x, y) in joints.items():
            world[i] = P(x * c, y, -x * s_)          # yaw about the vertical axis
            pixel[i] = P((w / 2 + px_per_m * x * c) / w, (h / 2 + px_per_m * y) / h, 0)
        return pixel, world

    for yaw in (0, 30, 60, 85):
        pixel, world = body(yaw)
        got = orientation_cos(pixel, world, 1280, 720)
        assert abs(got - abs(math.cos(math.radians(yaw)))) < 1e-6, (yaw, got)
    # Same pose, different frame shape: the cosine must not change.
    pixel_w, world_w = body(60, 1280, 720)
    pixel_s, world_s = body(60, 720, 720)
    assert abs(orientation_cos(pixel_w, world_w, 1280, 720)
               - orientation_cos(pixel_s, world_s, 720, 720)) < 1e-6

    # Every branch of the setup check, from synthetic poses.
    assert setup_check(None, None, 1280, 720)[0] is False
    assert setup_check(*body(80), 1280, 720) == (True, "Setup looks good.")
    ok, hint = setup_check(*body(0), 1280, 720)
    assert not ok and "side-on" in hint, hint
    assert setup_check(*body(0), 1280, 720, view="front")[0]
    pixel, world = body(80)
    pixel[27] = P(pixel[27].x, pixel[27].y, 0, v=0.1)
    assert "Whole body" in setup_check(pixel, world, 1280, 720)[1]
    pixel, world = body(80, px_per_m=420.0)               # feet run off the bottom
    assert "edge" in setup_check(pixel, world, 1280, 720)[1]

    print("analysis.py self-check passed")


if __name__ == "__main__":
    demo()
