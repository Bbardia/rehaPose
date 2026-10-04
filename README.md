# rehaPose

Live joint-angle biofeedback for rehab. Point a camera at yourself, press Start, do your
exercise, press Stop and read the results.

One window: camera with skeleton overlay on the left, eight live joint graphs on the
right, a results table when you stop.

## Install

Needs **Python 3.11 or newer**, on an Apple Silicon Mac, Windows x64 or Linux x64. Only
macOS arm64 is tested. Intel Macs: no `mediapipe==1.0.0` wheel. Windows and Linux on ARM:
no PyQt5 wheel, so pip tries to compile it from source. On Linux you may also need the Qt
xcb libraries (`libxcb-*`, `libxkbcommon-x11-0`) and GStreamer for the beeps, and the
user must be in the `video` group for the camera.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python rehapose.py
```

That is the whole install. Three dependencies, no compilation, no CUDA toolkit, no model
zoo. The two pose models (~6 MB and ~30 MB) download themselves to `~/.cache/rehapose/`
the first time you press Start — **the first run needs a network connection for that one
download**. Both come down up front, so a switch to the smaller one can never stall a
session on a download.
Everything after it runs offline. The model version is pinned and every model file is
checked against its published MD5, so the model cannot change underneath you between
sessions.

On first use rehaPose asks where to keep your sessions. Pick somewhere you can find and
back up; `~/Documents/rehaPose` is the default. Any sessions from an older version are
moved across automatically.

On macOS the first launch asks for camera permission. If you launched from a terminal,
grant it to the terminal in System Settings → Privacy & Security → Camera.

If the wrong camera opens: `python rehapose.py --camera 1`.

### Optional: the more accurate model (RTMPose, 2D)

```bash
.venv/bin/pip install onnxruntime==1.30.0 tqdm      # NVIDIA GPU: onnxruntime-gpu==1.30.0
.venv/bin/pip install --no-deps rtmlib==0.0.16      # --no-deps, see below
```

Then pick it under **Model → RTMPose**. It is OpenMMLab's RTMPose model family — from the
same toolbox as the original MMPose version — running through ONNX Runtime, so there is
still nothing to compile. `--no-deps` matters: rtmlib asks for `opencv-python`, which would install a second,
conflicting OpenCV next to the one MediaPipe brings (`pip check` will mention it; that is
expected). The first time it runs, it downloads its models once (about 275 MB), each pinned
and checked against a SHA-256 like the MediaPipe ones. It runs on an NVIDIA GPU (CUDA), on
an Apple Silicon GPU (CoreML) or on the CPU, and the Model menu says which it found.

## Use

Pick an exercise, press **Start** (or `Cmd+R`), move, press **Stop**. The session is saved
by itself and the numbers appear on the right.

- **30-second chair stand** — a scored test. Sit on a 43–45 cm chair against a wall,
  arms crossed at the chest, and stand fully and sit as many times as you can in 30
  seconds. Once a knee is in view the app counts down **3-2-1 with a tone on each
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
code, and a new code asks for both again (`age ?`, `sex ?`) rather than inheriting the
last person's — otherwise the previous patient's reference would print beside the next
patient's count. Until both are entered, a chair stand prints its count with no
reference.

**You do not need your whole body in frame** — only the joints the exercise uses: knees
for knee flexion, heel slides and the chair stand, shoulders for the shoulder exercises
(the shoulder angle is measured against your trunk, so keep that side's hip in view too).
Recording starts as soon as the camera has seen you for about a third of a second, and
the skeleton is drawn from the first frame, only where the model is confident.

Framing is **advice, not a gate**: the status line says when the exercise's joints are
out of frame or the camera angle is wrong for the movement — side-on for bending (knee,
hip and shoulder flexion, chair stand), facing the camera for abduction — and the results
report how much of the session was well framed. Only the scored chair stand waits, for a
knee to be in view, before its countdown. Camera placement is still the largest error
source you control, so take the advice when you can.

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

If the camera never saw you, nothing is written: an all-zeros row is not a session.
If the camera fails mid-session, what was recorded up to that point is kept and saved.
If the save itself fails (full disk, unplugged drive) the status line says **NOT SAVED**
and the results stay on screen. Until a copy exists the app will not let that session
go quietly: Start, Quit and opening a stored session all stop and ask first, and opening
History retries the save by itself once the folder is back.

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
mode. `python mirror_check.py clip.mp4 [--backend rtmpose]` runs the same test over every
frame of a recorded side-on clip, per model size, through the app's own model code — that
is the number to quote once a clip exists.

A first look at the two models, on a still clip of that same photo (median mirror gap):
MediaPipe heavy knee 3.3° / hip 11.4° / elbow 32.0° / shoulder 18.3°; RTMPose-x 1.5° /
1.5° / 2.2° / 1.5°. That says RTMPose places keypoints far more consistently. It does not
yet say its angles are more *accurate*: mirroring is an exact 2D operation, so this test
flatters 2D models, and a 2D angle is only right when the movement is square to the camera.

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

**Model** menu, saved between sessions and locked while recording:

- **MediaPipe — 3D** (default; always installed). Angles come from metric 3D landmarks, so
  they degrade gracefully when the camera is not square to the movement, and the app can
  tell you when you are turned the wrong way.
- **RTMPose — 2D, most accurate keypoints** (optional install above; the default on a
  machine with an NVIDIA GPU). Steadier keypoints, but its angles are measured **as the
  camera sees them**: correct when the movement is square to the camera — side-on for
  bending, facing it for abduction — and wrong when it is not. It has no depth, so it
  cannot warn you about the camera angle; framing advice is limited to what is in frame.

Every session file records which model (and which GPU or CPU) produced it, and History
has a Model column. Compare like with like: a 2D and a 3D angle of the same knee are not
the same number.

Within either model the app picks the size by **measuring itself**, not by inspecting
your hardware. MediaPipe has `heavy` and `lite`; RTMPose has `x`, `m` and `s` (all behind
one small person detector). Measured on an M4: RTMPose-x 31 ms per frame on the Apple GPU
and 88 ms on the CPU, `m` 32 ms on the CPU. The CUDA path has not been run on an NVIDIA
machine yet.

- Starts on the largest model (`heavy`, or RTMPose `x`). (Whether the larger MediaPipe
  model is the more *accurate* one is open: on the one mirror-test photo above, `lite` was
  more self-consistent on every joint. `mirror_check.py` on a real clip settles it.)
- For the first 5 seconds *after it first sees you*, it times its own inference.
- If the median is slower than 40 ms it drops a size and says so in the status bar.
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

So the default model is MediaPipe, with nothing to install, and the OpenMMLab models come
back as the optional RTMPose backend above: the same family, exported to ONNX by OpenMMLab
and run through rtmlib and ONNX Runtime, so nothing is compiled and nothing depends on
mmcv or torch.

## Checks

```bash
.venv/bin/pip install ruff==0.16.8   # once: the linter version CI pins
git config core.hooksPath githooks   # once per clone: the checks below run on commit and push

githooks/pre-commit                  # ~10 s, every commit: lint, maths self-check, UI smoke test
githooks/pre-push                    # ~35 s, every push: all of that, then the e2e scenarios
python mirror_check.py clip.mp4      # mirror consistency on a real clip (--backend rtmpose)
radon cc -s -a *.py                  # complexity
```

- **`test_rehapose.py`** runs real landmarks through the real UI with a stand-in for the
  camera thread, including the real setup check on an upper-body crop with both models.
- **`test_e2e.py`** runs the whole app — window, camera thread, model, signals, autosave,
  History — on clips generated from the test photo, clicking the real buttons. It covers
  the paths a regression would most likely break: recording while seated, an empty room,
  the full chair-stand flow, quitting mid-session, a first run offline, Stop during the
  model download, the heavy-to-lite switch, a real video file and a missing one.
- On commit and push the hooks test **exactly what is committed or pushed**, from a clean
  snapshot - a file you forgot to `git add` fails the commit, and unrelated work in progress
  does not. Run by hand, they test your working tree.
- Both keep everything in a temporary folder and never touch your real settings or camera.
  On failure, `REHAPOSE_ARTIFACTS=dir` saves a screenshot of the window and the sessions.
- CI runs all of it on macOS arm64 and Ubuntu x64 and uploads those artifacts on failure.
- **What only a real device can check** — the camera prompt, the tones you hear, a real
  chair stand — stays a manual run before each release.

## Files

| File | What |
|---|---|
| `rehapose.py` | The app: Qt UI, menus, history, autosave, chair-stand flow |
| `analysis.py` | Angles, filter, rep spans, setup check, exercises, norms. No Qt, no camera |
| `storage.py` | Sessions folder, legacy migration, the session file format and its readers |
| `capture.py` | Camera thread, tier ratchet wiring, pinned model download and hash check |
| `test_rehapose.py` | Headless smoke test: real landmarks through the real UI |
| `test_e2e.py` | End to end: the real app and camera thread on generated clips |
| `githooks/` | `pre-commit` and `pre-push` hooks that run the checks |
| `mirror_check.py` | Mirror-consistency measurement over a recorded clip, per model tier |
| `.github/workflows/checks.yml` | CI: all checks on macOS arm64 and Ubuntu x64, artifacts on failure |
| `Body_Joint.py` | The original MMPose version. Kept as reference; does not run — see above |
