"""Auto-straighten images using deskew (primary) or OpenCV Hough lines (fallback)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _rotate_pil(img: Image.Image, angle: float, expand: bool = True) -> Image.Image:
    """Rotate a PIL image, preserving transparency."""
    return img.rotate(angle, expand=expand, resample=Image.BICUBIC)


def straighten_deskew(input_path: Path, output_path: Path) -> tuple[Path, float]:
    """Detect and correct skew using the `deskew` library.

    Best for documents and product photos with clear edges.
    Returns (output_path, angle_applied).
    """
    from deskew import determine_skew
    from skimage import io as skio

    img_sk = skio.imread(str(input_path))
    angle = determine_skew(img_sk)

    if angle is None or abs(angle) < 0.1:
        # Nothing to do — copy as-is
        img = Image.open(input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        return output_path, 0.0

    img = Image.open(input_path).convert("RGBA")
    rotated = _rotate_pil(img, angle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rotated.save(output_path)
    return output_path, angle


def straighten_hough(input_path: Path, output_path: Path) -> tuple[Path, float]:
    """Detect rotation via OpenCV Hough line transform and correct it.

    More robust for product photos where there are no clear text lines.
    Uses pixel density analysis across candidate angles.
    """
    import cv2

    bgr = cv2.imread(str(input_path))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80,
        minLineLength=max(30, min(gray.shape) // 10),
        maxLineGap=20,
    )

    if lines is None or len(lines) == 0:
        img = Image.open(input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        return output_path, 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Normalise to (-45, 45]
        while angle <= -45:
            angle += 90
        while angle > 45:
            angle -= 90
        angles.append(angle)

    angle = float(np.median(angles))

    if abs(angle) < 0.3:
        img = Image.open(input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        return output_path, 0.0

    img = Image.open(input_path).convert("RGBA")
    rotated = _rotate_pil(img, angle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rotated.save(output_path)
    return output_path, angle


def straighten_image(
    input_path: Path,
    output_path: Path,
    method: str = "deskew",
) -> tuple[Path, float]:
    """Straighten an image.

    method: 'deskew' (default) — uses the deskew library (fast, good for most cases)
            'hough'             — OpenCV Hough lines (better for images with strong edges)
            'auto'              — tries deskew first; falls back to hough if angle is 0
    """
    if method == "hough":
        return straighten_hough(input_path, output_path)
    if method == "auto":
        path, angle = straighten_deskew(input_path, output_path)
        if abs(angle) < 0.1:
            return straighten_hough(input_path, output_path)
        return path, angle
    return straighten_deskew(input_path, output_path)
