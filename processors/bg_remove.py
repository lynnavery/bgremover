from pathlib import Path


# ── API backend (HuggingFace Space) ──────────────────────────────────────────

_client = None


def _get_client():
    global _client
    if _client is None:
        from gradio_client import Client
        _client = Client("not-lain/background-removal")
    return _client


def remove_background_api(input_path: Path, output_path: Path) -> Path:
    """Remove background via the not-lain/background-removal HuggingFace Space."""
    from gradio_client import handle_file
    import shutil

    client = _get_client()
    result = client.predict(
        f=handle_file(str(input_path)),
        api_name="/png",
    )
    result_path = Path(result[0]) if isinstance(result, (list, tuple)) else Path(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path != output_path:
        shutil.copy2(result_path, output_path)
    return output_path


# ── Local backend (rembg, runs on CPU) ───────────────────────────────────────

_rembg_sessions: dict = {}


def _get_rembg_session(model: str = "birefnet-portrait"):
    if model not in _rembg_sessions:
        from rembg import new_session
        _rembg_sessions[model] = new_session(model)
    return _rembg_sessions[model]


def remove_background_local(
    input_path: Path, output_path: Path, model: str = "birefnet-portrait"
) -> Path:
    """Remove background locally using rembg (no API quota, runs on CPU).

    model: rembg model name. Quality tiers:
      'birefnet-general'  — best open/commercial (~85% accuracy)  [default]
      'bria-rmbg'         — closest to Photoshop (~90%), non-commercial license
      'birefnet-portrait' — optimised for people
      'isnet-general-use' — solid all-rounder
      'u2net'             — original baseline (fastest, lowest quality)
    """
    from rembg import remove
    from PIL import Image

    session = _get_rembg_session(model)
    img = Image.open(input_path)
    result = remove(img, session=session)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    return output_path


# ── Dispatcher ────────────────────────────────────────────────────────────────

def remove_background(
    input_path: Path,
    output_path: Path,
    backend: str = "local",
    model: str = "birefnet-portrait",
) -> Path:
    """Remove background from an image.

    backend: 'local' — rembg (CPU, no quota limits)  [default]
             'api'   — HuggingFace Space (GPU, subject to free-tier quota)
    model:   rembg model name (only used when backend='local')
    """
    if backend == "api":
        return remove_background_api(input_path, output_path)
    return remove_background_local(input_path, output_path, model=model)
