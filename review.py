#!/usr/bin/env python3
"""review.py — interactive image review with kitten icat.

Usage:
    python review.py [OUTPUT_DIR]
    OUTPUT_DIR defaults to $BGREMOVER_OUTPUT or 'output'.

Controls:
    ←  / →      rotate –0.1° / +0.1°  (buffered: image redraws 1 s after last press)
    ⇧← / ⇧→    rotate –1°   / +1°
    Backspace   delete last rename character
    ESC         quit
    Enter       commit rename (if any) + save rotation (if any) → next image
    ⌥⌫          delete image
    any key     append to rename buffer
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from blessed import Terminal
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
ROTATE_STEP = 0.1   # degrees per arrow keypress
UI_LINES    = 5     # rows reserved below image for status / help / prompt / error


# ── image display ─────────────────────────────────────────────────────────────

def _kitten() -> str:
    return shutil.which("kitten") or "/Applications/kitty.app/Contents/MacOS/kitten"


_KITTY_DELETE_ALL = "\x1b_Ga=d;\x1b\\"   # graphics-protocol: delete all images


def _icat(path: Path, cols: int, rows: int) -> None:
    """Render image at the top-left corner of the terminal."""
    # Explicitly delete existing kitty graphics before drawing — term.clear
    # only erases text; without this, stale images conflict with the new one
    # and icat fails silently, leaving a blank screen.
    sys.stdout.write(_KITTY_DELETE_ALL)
    sys.stdout.flush()
    subprocess.run(
        [_kitten(), "icat",
         "--place", f"{cols}x{rows}@1x1",   # 1-indexed column × row
         "--scale-up",
         "--stdin", "no",
         str(path)],
        check=False,
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _rotated_path(original: Image.Image, angle: float, tmp: Path) -> Path:
    """Return tmp after saving a clockwise-rotated copy of original into it."""
    original.rotate(-angle, expand=True, resample=Image.BICUBIC).save(tmp)
    return tmp


def _save_rotation(src: Path, original: Image.Image, angle: float) -> None:
    """Overwrite src on disk with original rotated clockwise by angle degrees."""
    rotated = original.rotate(-angle, expand=True, resample=Image.BICUBIC)
    fmt = (original.format or "PNG").upper()
    kw = {"quality": 97, "subsampling": 0} if fmt in ("JPEG", "JPG") else {}
    rotated.save(src, format=fmt, **kw)


# ── main review loop ──────────────────────────────────────────────────────────

def review(output_dir: Path) -> None:
    images: list[Path] = sorted(
        f for f in output_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        print(f"No images found in {output_dir}")
        return

    term = Terminal()
    tmp  = Path(tempfile.mktemp(suffix=".png"))

    img_rows = max(term.height - UI_LINES, 10)
    img_cols = term.width
    ui_y     = img_rows   # 0-indexed first row of the UI panel

    # ── draw helpers ──────────────────────────────────────────────────────────

    def draw_image(src: Path, original: Image.Image, angle: float) -> None:
        sys.stdout.write(term.clear)
        sys.stdout.flush()
        path = _rotated_path(original, angle, tmp) if angle != 0.0 else src
        _icat(path, img_cols, img_rows)

    def draw_ui(idx: int, src: Path, angle: float, buf: str, err: str = "") -> None:
        angle_tag  = f"  [{angle:+.1f}°]" if angle else ""
        rename_tag = f"  →  {buf}_" if buf else ""
        s_status = (
            term.move(ui_y,   0) + term.clear_eol
            + term.bold + f"  [{idx+1}/{len(images)}]  {src.name}{angle_tag}{rename_tag}" + term.normal
        )
        s_help = (
            term.move(ui_y+1, 0) + term.clear_eol
            + term.dim + "  ←→ 0.1°  ·  ⇧←→ 1°  ·  Enter next  ·  ⌥⌫ delete  ·  ⌥ESC undo  ·  ESC quit  ·  type = rename" + term.normal
        )
        s_prompt = (
            term.move(ui_y+2, 0) + term.clear_eol
            + f"  > {buf}"
        )
        s_err = (
            term.move(ui_y+3, 0) + term.clear_eol
            + (term.red(f"  {err}") if err else "")
        )
        sys.stdout.write(s_status + s_help + s_prompt + s_err)
        sys.stdout.flush()

    # ── outer loop: one iteration per image ───────────────────────────────────

    try:
        with term.cbreak(), term.hidden_cursor():
            idx  = 0
            done = False

            while idx < len(images) and not done:
                src      = images[idx]
                original = Image.open(src).copy()
                angle    = 0.0
                buf      = ""
                err      = ""

                draw_image(src, original, angle)
                draw_ui(idx, src, angle, buf)

                # ── inner loop: keypresses for this image ─────────────────────
                _ROT = {
                    "KEY_LEFT":        -ROTATE_STEP,
                    "KEY_RIGHT":       +ROTATE_STEP,
                    "KEY_SLEFT":       -1.0,
                    "KEY_SRIGHT":      +1.0,
                    "KEY_SHIFT_LEFT":  -1.0,
                    "KEY_SHIFT_RIGHT": +1.0,
                }

                pending = None
                while True:
                    key = pending if pending is not None else term.inkey()
                    pending = None

                    if key.name in _ROT:
                        # ── rotation: accumulate presses, redraw after 1 s idle ─
                        angle = round(angle + _ROT[key.name], 1)
                        draw_ui(idx, src, angle, buf)   # cheap: angle in status

                        while True:
                            nk = term.inkey(timeout=1.0)
                            if nk and nk.name in _ROT:
                                angle = round(angle + _ROT[nk.name], 1)
                                draw_ui(idx, src, angle, buf)
                            else:
                                draw_image(src, original, angle)   # expensive redraw
                                draw_ui(idx, src, angle, buf)
                                if nk:
                                    pending = nk   # handle non-rotation key next
                                break

                    elif key.name in ("KEY_BACKSPACE", "KEY_DELETE"):
                        if buf:
                            buf = buf[:-1]
                            err = ""
                            draw_ui(idx, src, angle, buf)

                    elif key.name == "KEY_ALT_BACKSPACE" or str(key) == "\x1b\x7f":
                        src.unlink(missing_ok=True)
                        images.pop(idx)
                        break                   # idx stays; next image shifts in

                    elif key.name == "KEY_ALT_ESCAPE" or str(key) == "\x1b\x1b":
                        angle = 0.0
                        buf   = ""
                        err   = ""
                        draw_image(src, original, angle)
                        draw_ui(idx, src, angle, buf)

                    elif key.name == "KEY_ESCAPE":
                        done = True
                        break

                    elif key.name in ("KEY_ENTER", "KEY_RETURN") or str(key) in ("\r", "\n"):
                        # ── commit rotation + optional rename, then advance ────
                        if angle != 0.0:
                            _save_rotation(src, original, angle)

                        cmd = buf.strip()
                        if cmd:
                            new = src.with_stem(cmd)
                            if new.exists():
                                err = f"'{new.name}' already exists"
                                buf = ""
                                draw_ui(idx, src, angle, buf, err)
                                continue        # stay on this image
                            src.rename(new)
                            images[idx] = new

                        idx += 1
                        break                   # advance to next image

                    elif key.name is None:
                        # ── regular printable character ───────────────────────
                        ch = str(key)
                        if not ch:
                            continue

                        if ch.isprintable():
                            buf += ch
                            err  = ""
                            draw_ui(idx, src, angle, buf)

        sys.stdout.write(term.clear)
        sys.stdout.flush()

    except KeyboardInterrupt:
        sys.stdout.write(term.clear)
        sys.stdout.flush()

    finally:
        tmp.unlink(missing_ok=True)

    remaining = sum(1 for f in images if f.exists())
    print(f"Done. {remaining} image(s) in {output_dir}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("BGREMOVER_OUTPUT", "output"))
    if not d.is_dir():
        print(f"Directory not found: {d}", file=sys.stderr)
        sys.exit(1)
    review(d)
