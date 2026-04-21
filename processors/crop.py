"""Cropping via Florence-2 auto-crop (local) with fallback to content-bbox crop."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image


def crop_content_bbox(input_path: Path, output_path: Path, padding: int = 0) -> Path:
    """Crop to the bounding box of non-transparent / non-white pixels.

    Works best after background removal (transparent PNG).
    Falls back to a simple non-white pixel bbox for JPEGs.
    """
    img = Image.open(input_path).convert("RGBA")
    r, g, b, a = img.split()

    # Use alpha channel if meaningful, otherwise fall back to luminance
    if a.getextrema()[0] < 250:
        # Has transparency — crop to opaque region
        bbox = a.point(lambda p: 255 if p > 10 else 0).getbbox()
    else:
        # No alpha — crop to non-white region
        import numpy as np
        arr = np.array(img)
        mask = ~((arr[:, :, 0] > 240) & (arr[:, :, 1] > 240) & (arr[:, :, 2] > 240))
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            bbox = None
        else:
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            bbox = (int(cmin), int(rmin), int(cmax) + 1, int(rmax) + 1)

    if bbox is None:
        img.save(output_path)
        return output_path

    if padding:
        w, h = img.size
        bbox = (
            max(0, bbox[0] - padding),
            max(0, bbox[1] - padding),
            min(w, bbox[2] + padding),
            min(h, bbox[3] + padding),
        )

    cropped = img.crop(bbox)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(output_path)
    return output_path


def crop_florence2(
    input_path: Path,
    output_path: Path,
    florence2_dir: Path,
    prompt: str = "Watermark",
    object_aware: bool = True,
    crop_threshold: float = 15,
) -> Path:
    """Crop using the Wi-zz/florence-2-auto-crop local tool.

    Requires the florence-2-auto-crop repo cloned at `florence2_dir`.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(florence2_dir / "crop.py"),
        str(input_path),
        "-o", str(output_path.parent),
        "--prompt", prompt,
        "--crop-threshold", str(crop_threshold),
    ]
    if object_aware:
        cmd.append("--object-aware")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(florence2_dir))
    if result.returncode != 0:
        raise RuntimeError(f"florence-2-auto-crop failed: {result.stderr.strip()}")

    # florence-2-auto-crop writes to output_dir with the same filename
    produced = output_path.parent / input_path.name
    if produced != output_path and produced.exists():
        import shutil
        shutil.move(str(produced), output_path)

    return output_path


def crop_image(
    input_path: Path,
    output_path: Path,
    mode: str = "bbox",
    padding: int = 0,
    florence2_dir: Path | None = None,
    florence2_prompt: str = "Watermark",
    object_aware: bool = True,
    crop_threshold: float = 15,
) -> Path:
    """Dispatch to the appropriate crop implementation.

    mode: 'bbox'     — fast bounding-box crop (default)
          'florence2' — use local Florence-2 model (requires florence2_dir)
    """
    if mode == "florence2":
        if florence2_dir is None or not florence2_dir.exists():
            raise ValueError(
                "Florence-2 crop requires --florence2-dir pointing to the cloned "
                "Wi-zz/florence-2-auto-crop repository."
            )
        return crop_florence2(
            input_path, output_path, florence2_dir,
            prompt=florence2_prompt,
            object_aware=object_aware,
            crop_threshold=crop_threshold,
        )
    return crop_content_bbox(input_path, output_path, padding=padding)
