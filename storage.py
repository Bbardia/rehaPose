"""Session storage: folder choice, ~/.cache migration, and the session CSV layout."""
import csv
import datetime
import os
import shutil
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

# Pre-chooser location under the model cache, which the OS may purge.
LEGACY_SESSIONS = Path.home() / ".cache" / "rehapose" / "sessions"
JOINT_COLUMNS = ["joint", "rom_deg", "peak_deg", "min_deg", "reps", "tracked_pct"]


def settings():
    # No args: use the QApplication's names, so tests never touch the real dataDir.
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
        return None   # unplugged drive, revoked permission, or path became a file
    return path


def choose_data_dir(parent):
    """Ask once where sessions live (the user's only copy), then remember it."""
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
        # Keep the missing folder: falling back would split sessions across two folders.
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
    """Move sessions out of ~/.cache, which the OS may delete."""
    if not LEGACY_SESSIONS.is_dir():
        return 0
    target.mkdir(parents=True, exist_ok=True)
    moved = 0
    for old in LEGACY_SESSIONS.glob("*.csv"):
        new = target / old.name
        if new.exists():
            continue
        # Copy then os.replace: rename() fails across volumes, move() can leave a stub.
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
    # Spreadsheet re-saves pad blank rows with commas.
    return not any(cell.strip() for cell in row)


def read_header(path):
    """Block 1 of a session file as a dict; needs csv.reader on newline="" (CRLF)."""
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
    """Block 2 shaped like summarize(); raises ValueError/KeyError on a mangled file."""
    return {joint: {"rom": float(v["rom_deg"]), "peak": float(v["peak_deg"]),
                    "min": float(v["min_deg"]), "reps": int(v["reps"]),
                    "coverage": float(v["tracked_pct"])}
            for joint, v in read_joints(path).items()}


def new_session_path(target, key):
    """A fresh filename: timestamp first, so History sorts and stamp_of() parses it."""
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path, n = target / f"{stamp}-{key}.csv", 2
    while path.exists():           # same-second stops must not overwrite
        path, n = target / f"{stamp}-{key}-{n}.csv", n + 1
    return path


def write_session(path, header, summaries, times, angles):
    """Header (max two pairs per row), per-joint summary, per-frame trace; blank-row separated."""
    joints = list(summaries)
    path = Path(path)
    # Side name + swap: a full disk must not leave a truncated session History would list.
    part = path.with_name(path.name + ".part")
    try:
        _write(part, header, summaries, times, angles, joints)
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)


def _write(part, header, summaries, times, angles, joints):
    # utf-8-sig: the BOM makes Excel show the degree sign.
    with open(part, "w", newline="", encoding="utf-8-sig") as handle:
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
