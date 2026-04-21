#!/usr/bin/env python3
"""bgremover — batch CLI image editor.

Supports background removal, auto-crop, and auto-straighten.
"""

from __future__ import annotations

import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
from yaspin import yaspin
from yaspin.spinners import Spinners

_SP = Spinners.arc


def _spin(text: str):
    return yaspin(_SP, text=text, color="cyan")


def _err(msg: str) -> None:
    click.echo(msg, err=True)


def apply_memory_limit(limit_gb: float) -> None:
    import math, os

    headroom_gb = limit_gb * 0.25
    max_threads = max(1, math.floor(headroom_gb / 0.5))
    threads = min(max_threads, os.cpu_count() or 4)

    try:
        import onnxruntime as ort
        _orig = ort.SessionOptions

        class _MemSafe(_orig):
            def __init__(self):
                super().__init__()
                self.enable_mem_pattern = False
                self.enable_cpu_mem_arena = False
                self.intra_op_num_threads = threads
                self.inter_op_num_threads = threads

        ort.SessionOptions = _MemSafe
    except ImportError:
        pass

    try:
        import torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(threads)
    except (ImportError, RuntimeError):
        pass


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
HEIC_EXTS  = {".heic", ".heif"}


def collect_images(paths: tuple[str, ...], recursive: bool) -> list[Path]:
    images: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            if p.suffix.lower() in IMAGE_EXTS | HEIC_EXTS:
                images.append(p)
        elif p.is_dir():
            glob = p.rglob("*") if recursive else p.glob("*")
            images.extend(
                f for f in glob
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS | HEIC_EXTS
            )
        else:
            _err(f"Warning: {raw!r} not found, skipping.")
    return sorted(set(images))


def convert_heic(src: Path, dest_dir: Path) -> Path:
    import shutil, subprocess
    if not shutil.which("magick"):
        raise RuntimeError("ImageMagick 'magick' not found — brew install imagemagick")
    dest_dir.mkdir(parents=True, exist_ok=True)
    jpg = dest_dir / src.with_suffix(".jpg").name
    result = subprocess.run(["magick", str(src), str(jpg)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"magick failed for {src}: {result.stderr.strip()}")
    return jpg


def maybe_convert_heic(images: list[Path], tmp_dir: Path, sp) -> list[Path]:
    heic = [p for p in images if p.suffix.lower() in HEIC_EXTS]
    if not heic:
        return images

    converted: dict[Path, Path] = {}
    errors: list[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(convert_heic, p, tmp_dir): p for p in heic}
        for future in as_completed(futures):
            src = futures[future]
            done += 1
            try:
                converted[src] = future.result()
            except Exception as exc:
                errors.append(f"{src}: {exc}")
            sp.text = f"convert HEIC  {done}/{len(heic)}"

    for e in errors:
        sp.write(f"  HEIC error: {e}")

    return [converted.get(p, p) for p in images]


def resolve_output(src: Path, input_root: Path | None, output_dir: Path, suffix: str = "") -> Path:
    try:
        rel = src.relative_to(input_root) if input_root else src.name
    except ValueError:
        rel = src.name
    out = output_dir / rel
    if suffix:
        out = out.with_suffix(suffix)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Shared options
# ──────────────────────────────────────────────────────────────────────────────

_input_args   = click.argument("inputs", nargs=-1, required=True, metavar="IMAGE_OR_DIR...")
_output_opt   = click.option("-o", "--output", "output_dir", default="output", show_default=True,
                             envvar="BGREMOVER_OUTPUT",
                             help="Directory for processed images.")
_recursive_opt = click.option("-r", "--recursive", is_flag=True,
                               help="Recurse into subdirectories.")
_workers_opt  = click.option("-w", "--workers", default=4, show_default=True,
                              envvar="BGREMOVER_WORKERS", help="Parallel worker threads.")
_overwrite_opt = click.option("--overwrite/--no-overwrite", default=False, show_default=True,
                               help="Overwrite existing output files.")
_backend_opt  = click.option("--backend", type=click.Choice(["local", "api"]), default="local",
                              show_default=True, envvar="BGREMOVER_BACKEND")
_model_opt    = click.option(
    "--model",
    type=click.Choice(["birefnet-general", "birefnet-portrait", "bria-rmbg",
                       "isnet-general-use", "u2net"]),
    default="birefnet-portrait", show_default=True, envvar="BGREMOVER_MODEL",
)


@click.group()
@click.option("--max-memory", default=12.0, show_default=True, type=float, metavar="GB",
              envvar="BGREMOVER_MAX_MEMORY", help="Cap process memory in GB.")
@click.pass_context
def cli(ctx: click.Context, max_memory: float) -> None:
    """Batch image processor: remove backgrounds, crop, and straighten images."""
    ctx.ensure_object(dict)
    apply_memory_limit(max_memory)


# ──────────────────────────────────────────────────────────────────────────────
# remove-bg
# ──────────────────────────────────────────────────────────────────────────────

@cli.command("remove-bg")
@_input_args
@_output_opt
@_recursive_opt
@_workers_opt
@_overwrite_opt
@_backend_opt
@_model_opt
def remove_bg_cmd(inputs, output_dir, recursive, workers, overwrite, backend, model):
    """Remove backgrounds from images."""
    from processors.bg_remove import remove_background

    images = collect_images(inputs, recursive)
    if not images:
        _err("No images found."); sys.exit(1)

    out_root   = Path(output_dir)
    input_root = Path(inputs[0]) if len(inputs) == 1 and Path(inputs[0]).is_dir() else None
    errors: list[str] = []

    with _spin("remove-bg") as sp:
        with tempfile.TemporaryDirectory() as heic_tmp:
            images = maybe_convert_heic(images, Path(heic_tmp), sp)
            total, done = len(images), 0
            sp.text = f"remove-bg  0/{total}"

            def _process(src: Path) -> tuple[Path, str | None]:
                dst = resolve_output(src, input_root, out_root, suffix=".png")
                if dst.exists() and not overwrite:
                    return dst, "skipped"
                try:
                    remove_background(src, dst, backend=backend, model=model)
                    return dst, None
                except Exception as exc:
                    return dst, str(exc)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_process, img): img for img in images}
                for future in as_completed(futures):
                    src = futures[future]
                    _, err = future.result()
                    done += 1
                    sp.text = f"remove-bg  {done}/{total}  {src.name}"
                    if err and err != "skipped":
                        errors.append(f"{src}: {err}")
                        sp.write(f"  ✗ {src.name}: {err}")

        sp.ok("✓") if not errors else sp.fail("✗")

    if errors:
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# crop
# ──────────────────────────────────────────────────────────────────────────────

@cli.command("crop")
@_input_args
@_output_opt
@_recursive_opt
@_workers_opt
@_overwrite_opt
@click.option("--mode", type=click.Choice(["bbox", "florence2"]), default="bbox", show_default=True)
@click.option("--padding", default=0, show_default=True)
@click.option("--florence2-dir", default=None, metavar="DIR")
@click.option("--florence2-prompt", default="Watermark", show_default=True)
@click.option("--object-aware/--no-object-aware", default=True, show_default=True)
@click.option("--crop-threshold", default=15.0, show_default=True)
def crop_cmd(inputs, output_dir, recursive, workers, overwrite,
             mode, padding, florence2_dir, florence2_prompt, object_aware, crop_threshold):
    """Crop images to content bounds (or via Florence-2 model)."""
    from processors.crop import crop_image

    f2_dir = Path(florence2_dir) if florence2_dir else None
    images = collect_images(inputs, recursive)
    if not images:
        _err("No images found."); sys.exit(1)

    out_root   = Path(output_dir)
    input_root = Path(inputs[0]) if len(inputs) == 1 and Path(inputs[0]).is_dir() else None
    errors: list[str] = []

    with _spin("crop") as sp:
        with tempfile.TemporaryDirectory() as heic_tmp:
            images = maybe_convert_heic(images, Path(heic_tmp), sp)
            total, done = len(images), 0
            sp.text = f"crop  0/{total}"

            def _process(src: Path) -> tuple[Path, str | None]:
                dst = resolve_output(src, input_root, out_root)
                if dst.exists() and not overwrite:
                    return dst, "skipped"
                try:
                    crop_image(src, dst, mode=mode, padding=padding, florence2_dir=f2_dir,
                               florence2_prompt=florence2_prompt, object_aware=object_aware,
                               crop_threshold=crop_threshold)
                    return dst, None
                except Exception as exc:
                    return dst, str(exc)

            actual_workers = 1 if mode == "florence2" else workers
            with ThreadPoolExecutor(max_workers=actual_workers) as pool:
                futures = {pool.submit(_process, img): img for img in images}
                for future in as_completed(futures):
                    src = futures[future]
                    _, err = future.result()
                    done += 1
                    sp.text = f"crop  {done}/{total}  {src.name}"
                    if err and err != "skipped":
                        errors.append(f"{src}: {err}")
                        sp.write(f"  ✗ {src.name}: {err}")

        sp.ok("✓") if not errors else sp.fail("✗")

    if errors:
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# straighten
# ──────────────────────────────────────────────────────────────────────────────

@cli.command("straighten")
@_input_args
@_output_opt
@_recursive_opt
@_workers_opt
@_overwrite_opt
@click.option("--method", type=click.Choice(["deskew", "hough", "auto"]), default="deskew",
              show_default=True)
def straighten_cmd(inputs, output_dir, recursive, workers, overwrite, method):
    """Auto-straighten / deskew images."""
    from processors.straighten import straighten_image

    images = collect_images(inputs, recursive)
    if not images:
        _err("No images found."); sys.exit(1)

    out_root   = Path(output_dir)
    input_root = Path(inputs[0]) if len(inputs) == 1 and Path(inputs[0]).is_dir() else None
    errors: list[str] = []

    with _spin("straighten") as sp:
        with tempfile.TemporaryDirectory() as heic_tmp:
            images = maybe_convert_heic(images, Path(heic_tmp), sp)
            total, done = len(images), 0
            sp.text = f"straighten  0/{total}"

            def _process(src: Path) -> tuple[Path, float, str | None]:
                dst = resolve_output(src, input_root, out_root)
                if dst.exists() and not overwrite:
                    return dst, 0.0, "skipped"
                try:
                    path, angle = straighten_image(src, dst, method=method)
                    return path, angle, None
                except Exception as exc:
                    return dst, 0.0, str(exc)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_process, img): img for img in images}
                for future in as_completed(futures):
                    src = futures[future]
                    _, angle, err = future.result()
                    done += 1
                    sp.text = f"straighten  {done}/{total}  {src.name}"
                    if err and err != "skipped":
                        errors.append(f"{src}: {err}")
                        sp.write(f"  ✗ {src.name}: {err}")
                    elif not err and abs(angle) > 0.1:
                        sp.write(f"  {src.name}  {angle:+.2f}°")

        sp.ok("✓") if not errors else sp.fail("✗")

    if errors:
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# process  (pipeline: straighten → remove-bg → crop)
# ──────────────────────────────────────────────────────────────────────────────

@cli.command("process")
@_input_args
@_output_opt
@_recursive_opt
@_workers_opt
@_overwrite_opt
@click.option("--remove-bg/--no-remove-bg", default=True, show_default=True)
@click.option("--crop/--no-crop", "do_crop", default=True, show_default=True)
@click.option("--straighten/--no-straighten", "do_straighten", default=False, show_default=True)
@click.option("--straighten-method", type=click.Choice(["deskew", "hough", "auto"]),
              default="deskew", show_default=True)
@click.option("--crop-mode", type=click.Choice(["bbox", "florence2"]), default="bbox",
              show_default=True)
@click.option("--crop-padding", default=0, show_default=True)
@click.option("--florence2-dir", default=None, metavar="DIR")
@click.option("--florence2-prompt", default="Watermark", show_default=True)
@click.option("--object-aware/--no-object-aware", default=True, show_default=True)
@click.option("--crop-threshold", default=15.0, show_default=True)
@_backend_opt
@_model_opt
@click.option("--review/--no-review", "do_review", default=True, show_default=True,
              help="Launch interactive review after processing.")
@click.option("--delete-originals/--no-delete-originals", default=True, show_default=True,
              help="Delete original input files after successful processing.")
def process_cmd(inputs, output_dir, recursive, workers, overwrite,
                remove_bg, do_crop, do_straighten, straighten_method,
                crop_mode, crop_padding, florence2_dir, florence2_prompt,
                object_aware, crop_threshold, backend, model,
                do_review, delete_originals):
    """Run the full pipeline: [straighten →] [remove-bg →] [crop] on each image.

    By default runs remove-bg + crop, then launches review and deletes originals.
    """
    import shutil
    from processors.bg_remove import remove_background
    from processors.crop import crop_image
    from processors.straighten import straighten_image

    f2_dir         = Path(florence2_dir) if florence2_dir else None
    original_images = collect_images(inputs, recursive)
    if not original_images:
        _err("No images found."); sys.exit(1)

    out_root   = Path(output_dir)
    input_root = Path(inputs[0]) if len(inputs) == 1 and Path(inputs[0]).is_dir() else None
    errors: list[str] = []
    succeeded: list[Path] = []
    heic_tmp = tempfile.mkdtemp()

    with _spin("process") as sp:
        try:
            images = maybe_convert_heic(list(original_images), Path(heic_tmp), sp)
            total, done = len(images), 0
            sp.text = f"process  0/{total}"

            def _process(src: Path, orig: Path) -> tuple[Path, str | None]:
                dst = resolve_output(src, input_root, out_root, suffix=".png")
                if dst.exists() and not overwrite:
                    return dst, "skipped"
                with tempfile.TemporaryDirectory() as tmp:
                    current, step = src, 0
                    try:
                        if do_straighten:
                            p = Path(tmp) / f"step{step}{src.suffix}"
                            straighten_image(current, p, method=straighten_method)
                            current, step = p, step + 1
                        if remove_bg:
                            p = Path(tmp) / f"step{step}.png"
                            remove_background(current, p, backend=backend, model=model)
                            current, step = p, step + 1
                        if do_crop:
                            p = Path(tmp) / f"step{step}.png"
                            crop_image(current, p, mode=crop_mode, padding=crop_padding,
                                       florence2_dir=f2_dir, florence2_prompt=florence2_prompt,
                                       object_aware=object_aware, crop_threshold=crop_threshold)
                            current = p
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(current, dst)
                        return dst, None
                    except Exception as exc:
                        return dst, str(exc)

            actual_workers = 1 if crop_mode == "florence2" else workers
            with ThreadPoolExecutor(max_workers=actual_workers) as pool:
                futures = {pool.submit(_process, img, orig): orig
                           for img, orig in zip(images, original_images)}
                for future in as_completed(futures):
                    orig = futures[future]
                    _, err = future.result()
                    done += 1
                    sp.text = f"process  {done}/{total}  {orig.name}"
                    if err and err != "skipped":
                        errors.append(f"{orig}: {err}")
                        sp.write(f"  ✗ {orig.name}: {err}")
                    elif not err:
                        succeeded.append(orig)
        finally:
            shutil.rmtree(heic_tmp, ignore_errors=True)

        sp.ok("✓") if not errors else sp.fail("✗")

    if delete_originals and succeeded:
        for orig in succeeded:
            orig.unlink(missing_ok=True)
        click.echo(f"Deleted {len(succeeded)} original(s).")

    if errors:
        sys.exit(1)

    if do_review:
        import subprocess as _sp
        _sp.run([sys.executable, str(Path(__file__).parent / "review.py"), output_dir])


if __name__ == "__main__":
    cli()
