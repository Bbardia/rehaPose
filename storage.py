"""Where sessions live and how a session file is laid out.

The user's only copy of every session goes through here, so this is the layer that must
never lose data: folder choice and its failure modes, the one-time migration out of
~/.cache, and reading files back after a spreadsheet has had its way with them.
No camera, no mediapipe; QtCore for QSettings and QtWidgets only for the folder picker.
"""
import csv
import datetime
import os
import shutil
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

# Sessions were once written under the model cache, which the OS may purge.
LEGACY_SESSIONS = Path.home() / ".cache" / "rehapose" / "sessions"
JOINT_COLUMNS = ["joint", "rom_deg", "peak_deg", "min_deg", "reps", "tracked_pct"]


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


def stored_summaries(path):
    """Block 2 in the shape summarize() returns, so one renderer serves live and stored.
    Raises ValueError/KeyError on a file that has been edited out of shape."""
    return {joint: {"rom": float(v["rom_deg"]), "peak": float(v["peak_deg"]),
                    "min": float(v["min_deg"]), "reps": int(v["reps"]),
                    "coverage": float(v["tracked_pct"])}
            for joint, v in read_joints(path).items()}


def new_session_path(target, key):
    """A fresh filename: timestamp first, so History sorts and stamp_of() parses it."""
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path, n = target / f"{stamp}-{key}.csv", 2
    while path.exists():           # two stops in one second must not overwrite
        path, n = target / f"{stamp}-{key}-{n}.csv", n + 1
    return path


def write_session(path, header, summaries, times, angles):
    """Three blocks separated by blank rows: header key/value rows (two pairs per row at
    most, which is what read_header understands), per-joint summary, per-frame trace."""
    joints = list(summaries)
    # utf-8-sig: the BOM is what makes Excel show the degree sign instead of mojibake.
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerows(header)
        writer.writerow([])
        writer.writerow(JOINT_COLUMNS)
        for joint, s in summaries.items():
            writer.writerow([joint, f"{s['rom']:.1f}", f"{s['peak']:.1f}",
                             f"{s['min']:.1f}", s["reps"], f"{s['coverage']:.1f}"])
        writer.writerow([])
        writer.writerow(["time_s"] + joints)
        for i, t in enumerate(times[joints[0]]):
            writer.writerow([f"{t:.3f}"] + ["" if angles[j][i] is None else f"{angles[j][i]:.2f}"
                                            for j in joints])
