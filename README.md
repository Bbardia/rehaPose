# rehaPose

Live joint-angle biofeedback for rehab. Point a camera at yourself, press Start, do your
exercise, press Stop and read the results.

One window: camera with skeleton overlay on the left, eight live joint graphs on the
right, a results table when you stop.

## Install

Needs **Python 3.11 or newer**, on an Apple Silicon Mac, Windows (x64 or arm64) or Linux
(x64 or arm64). Intel Macs are not supported: `mediapipe==1.0.0` ships no wheel for them,
and the version pin is not negotiable (see `requirements.txt`).

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python rehapose.py
```

That is the whole install. Three dependencies, no compilation, no CUDA toolkit, no model
zoo. The pose model (~6 MB or ~30 MB) downloads itself to `~/.cache/rehapose/` the first
time you press Start — **the first run needs a network connection for that one download**.
Everything after it runs offline. The model version is pinned and the download is checked
against its published MD5, so the model cannot change underneath you between sessions.

On first use rehaPose asks where to keep your sessions. Pick somewhere you can find and
back up; `~/Documents/rehaPose` is the default. Any sessions from an older version are
moved across automatically.

On macOS the first launch asks for camera permission. If you launched from a terminal,
grant it to the terminal in System Settings → Privacy & Security → Camera.

If the wrong camera opens: `python rehapose.py --camera 1`.

## Use

Pick an exercise, press **Start** (or `Cmd+R`), move, press **Stop**. The session is saved
by itself and the numbers appear on the right.

- **30-second chair stand** — a scored test. Sit on a 43–45 cm chair against a wall,
  arms crossed at the chest, and stand fully and sit as many times as you can in 30
  seconds. Once the framing is good the app counts down **3-2-1 with a tone on each
  number**, sounds a high tone for go and a low one when time is up, and shows the count
  large on the video. As the published protocol does, a final stand **more than halfway
  up** when time is called counts. It prints the published reference for the age and
  sex entered. Stop it early and you get the count but no reference — a 12-second count
  is not a 30-second score. It cannot verify chair height or that the arms stayed
  crossed, so the number is only as good as your protocol.
- **Everything else** — knee flexion, hip abduction, shoulder abduction and flexion,
  heel slides, or unlabelled. Start, move, Stop.

**Person code.** Type initials or a code (not a name) in the field next to the exercise.
It is written into every session and shown in History. Age and sex are remembered per
code, and a new code asks for the age again (`age ?`) rather than inheriting the last
person's — otherwise the previous patient's reference would print beside the next
patient's count.

In every mode the app will not start measuring until the setup is good. It tells you
what is wrong — side-on, whole body in frame, not clipping the edge — and starts the
clock the moment you are in position, not when you press the button. That matters:
camera placement is the largest error source you actually control.

Graphs show a rolling 20 seconds; the full session is kept and analysed on Stop. A gap
in a line means that joint was not confidently tracked in those frames.

**Beep on rep** gives an audible cue on each counted rep, so you can keep your eyes off
the screen — which is most of the point when you are on the floor.

## Results

On Stop, per joint:

| Column | Meaning |
|---|---|
| ROM | The **median** of the per-rep ranges, not the session's largest-minus-smallest |
| Peak / Min | The extremes reached |
| Reps | Completed out-and-back movements, by hysteresis on the joint's own range |
| Tracked | Share of frames where the joint was actually visible |

ROM is a median of reps because the naive largest-minus-smallest is destroyed by a
single bad frame. One 148° glitch in a clean 90° session reports 148° that way, and 90°
this way.

**Rows below 80% tracked report `--` instead of numbers.** A range computed from a third
of the frames is not a measurement, and printing it anyway is how a tool loses trust.

Every session is **saved automatically** on Stop — summary plus the full per-frame trace —
into the folder you chose on first run. **File → Save a Copy…** (`Cmd+S`) puts one
somewhere else as well, and **File → Show Sessions Folder** opens the folder.

If the framing never came good, nothing is written: an all-zeros row is not a session.
If the camera fails mid-session, what was recorded up to that point is kept and saved.
If the save itself fails (full disk, unplugged drive) the status line says **NOT SAVED**
and the results stay on screen — use **Save a Copy** before starting another session.

Each file also records the person code, app version, model tier (`heavy`/`lite`) and
mediapipe version, so a later analysis can tell which model produced which angles. Files
are UTF-8 with a BOM, which is what makes Excel show the degree sign correctly.

## History

**View → History** (`Cmd+2`) lists every session — date, person, exercise, length, headline
result and how much of it was well framed. The headline result names its joint ("right
knee 92°"): it is the largest reported ROM of all eight, which is not always the joint the
exercise is about. Double-click a session to load its numbers back into the results table;
a chair stand also brings back its count and reference. A file that has been mangled by
a spreadsheet re-save costs its own row, not the whole page.

Compare like with like: the same exercise, recorded from the same camera position. Treat
differences under about 10° as noise rather than progress. There is deliberately **no
trend line yet** — with the per-joint errors below, a band wide enough to be honest would
swallow everything it could show. It comes back when a repeatability run says what the
real noise floor is.

Angles use the clinical convention: **0° is full extension**. A straight leg is 0° knee
flexion, not 180°. Shoulders are the exception — 0° is the arm at your side.

## What this measures well, and what it does not

Measured on this code, with the heavy model, by mirroring one frame — flipping the image
swaps the person's left and right, so a perfect pipeline would report identical angles
and the gap is its own inconsistency:

| Joint | heavy | lite |
|---|---|---|
| Knee | 8.6° | 6.6° |
| Hip | 22.2° | 16.4° |
| Shoulder | 16.1° | 4.0° |
| Elbow | 54.4° | 3.9° |

That is one photograph, not a validation study, and the arms in it are ambiguous — which
is exactly why the elbow number is what it is. Treat it as a floor on the error, not a
spec. It is also front-on, in single-image mode, while the app records side-on in video
mode. `python mirror_check.py clip.mp4` runs the same test over every frame of a
recorded side-on clip, per tier, in the mode the app actually uses — that is the number
to quote once a clip exists.

**What it is fair to say:** counts and durations are dependable, because a chair stand
only requires detecting that a large knee excursion crossed a threshold, not knowing
where the knee was. Within-subject ROM **trends from the same camera position** are
usable; treat changes under about 10° as noise. Sagittal knee and hip are the
best-conditioned measurements here.

**What it is not fair to say:** that this replaces a goniometer. A goniometer's own
minimal significant difference is roughly 6–14°, and the errors above sit at or beyond
that. Do not read joint rotation, and do not read knee valgus from it — those need
either multiple calibrated cameras or a frontal setup this does not validate.

## Scope

rehaPose **measures and records exercise performance**. It does not assess, screen,
diagnose or evaluate anything, and it does not estimate fall risk. It reports a count
and, where one is published, the reference value beside it — the interpretation is the
clinician's. That distinction is also the line between a wellness tool and a regulated
medical device under EU MDR and Swiss MedDO, so the wording in the UI is deliberate.

Video is processed on your machine and never leaves it. Nothing is uploaded, there is no
account, and the only thing written to disk is joint angles.

## Keyboard

| | |
|---|---|
| `Cmd+R` | Start / Stop |
| `Cmd+1` / `Cmd+2` | Live / History |
| `Cmd+S` | Save a copy of the last session |
| `Cmd+?` | How to record |
| `Cmd+Q` | Quit — stops and saves a running session first |

## Tracked joints

Left and right knee, hip, elbow and shoulder.

## How the backend is chosen

There is one pose library (MediaPipe) with two model sizes, and the app picks between
them by **measuring itself**, not by inspecting your hardware:

- Starts on `heavy`, the larger model. (Whether it is the more *accurate* one is open:
  on the one mirror-test photo above, `lite` was more self-consistent on every joint.
  `mirror_check.py` on a real clip is what settles the default.)
- For the first 5 seconds *after it first sees you*, it times its own inference.
- If the median is slower than 40 ms it drops to `lite` and says so in the status bar.
- After that the choice is locked for the session.

It only ever ratchets downward, and only in the first few seconds, because switching
models mid-set shifts the measured angles slightly — a step change the rep counter would
read as real movement. On a modern machine the ratchet never fires and you silently get
the accurate model.

Timing only counts frames where a person was actually found: MediaPipe skips most of its
work on an empty frame, so measuring an empty room reads about 2.6× too optimistic and
would pick a model the machine cannot sustain.

## Where MMPose went

Earlier versions of this project used MMPose (`Body_Joint.py`, kept for reference but
no longer runnable). Automating that install was the goal; it turned out not to be
possible:

- `mmcv` ships only an sdist on PyPI, so plain `pip install` compiles C++/CUDA extensions
  against your exact installed torch.
- The prebuilt-wheel index is keyed by torch version and stops at torch 2.4; there has
  never been a macOS arm64 wheel past torch 2.1 CPU.
- The pins are mutually unsatisfiable: `mmpose` wants `mmdet<3.3.0`, `mmdet` wants
  `mmcv<2.2.0`, and `mim install mmcv` gives you 2.2.0.
- The stack is unmaintained — last releases were in 2024.

The evidence was already in this repo: the old `.venv` had `mmcv` installed but its
compiled `_ext` missing, which is why `Body_Joint.py` never ran on this machine.

So the default model is now MediaPipe `heavy`, auto-selected, with nothing to install.
If you ever need RTMPose or ViTPose specifically, the escape hatch is one line —
`pip install rtmlib` — which pulls them as ONNX with no compilation. It is not used here
because it returns 2D pixel keypoints only, and this app measures angles from **metric 3D
world landmarks**, which degrade more gracefully off-axis. They do still degrade, which is
why there is a setup gate and why the numbers above are what they are.

## Checks

```bash
python analysis.py                                  # angle, filter, rep-span, ratchet asserts
QT_QPA_PLATFORM=offscreen python test_rehapose.py   # real landmarks through the real UI
ruff check .                                        # lint (config in ruff.toml)
python mirror_check.py clip.mp4                     # mirror consistency on a real clip
radon cc -s -a rehapose.py analysis.py              # complexity
```

## Files

| File | What |
|---|---|
| `rehapose.py` | The app: camera thread, tier ratchet, Qt UI, menus, history, autosave |
| `analysis.py` | Angles, filter, rep spans, setup check, exercises, norms. No Qt, no camera |
| `test_rehapose.py` | Headless smoke test |
| `mirror_check.py` | Mirror-consistency measurement over a recorded clip, per model tier |
| `.github/workflows/checks.yml` | CI: lint, self-check and smoke test on macOS arm64 |
| `Body_Joint.py` | The original MMPose version. Kept as reference; does not run — see above |
