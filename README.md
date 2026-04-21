# bgremover

Batch image processor for removing backgrounds, auto-cropping, and straightening images. Includes a browser-based UI and a CLI.

## Web UI

Open `web/index.html` in a browser (via a local server — required for the web worker).

- Drop images or a ZIP of images onto the drop zone
- Supports JPEG, PNG, WebP, HEIC/HEIF
- Background removal runs locally in-browser via [RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4) (Transformers.js)
- Review page: rename files, rotate, download individually or as a ZIP

## CLI

```
pip install -r requirements.txt
```

```
python cli.py [--max-memory GB] <command> [options] <inputs...>
```

### Commands

**`remove-bg`** — remove image backgrounds

```
python cli.py remove-bg [--output DIR] [--backend onnx|rembg] [--model MODEL] <images or dirs>
```

**`crop`** — auto-crop to subject

```
python cli.py crop [--mode bbox|florence2] [--padding PX] [--florence2-dir DIR] <images or dirs>
```

**`straighten`** — deskew/straighten images

```
python cli.py straighten [--method deskew|hough|auto] <images or dirs>
```

**`process`** — run all steps in one pass

```
python cli.py process [--remove-bg/--no-remove-bg] [--crop/--no-crop] [--straighten/--no-straighten] <images or dirs>
```

### Common options

| Flag | Default | Description |
|------|---------|-------------|
| `--output DIR` | `./output` | Output directory |
| `--recursive` | off | Recurse into subdirectories |
| `--workers N` | 4 | Parallel workers |
| `--overwrite` | off | Overwrite existing output files |
| `--max-memory GB` | 12 | Cap process memory usage |

## Florence-2 crop

The `florence2` crop mode uses a local [Florence-2](https://huggingface.co/Wi-zz/florence-2-auto-crop) model checkpoint for object-aware cropping. Point `--florence2-dir` at the model directory.

```
git submodule update --init
python cli.py crop --mode florence2 --florence2-dir florence-2-auto-crop <images>
```
