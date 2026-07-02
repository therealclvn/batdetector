# Bat Model 3

Thermal bat detection, tracking, trajectory visualization, and Excel report export.

## Directory Layout

```text
models/             Current model files
models/archive/     Archived model files
videos/             Videos currently being analyzed
videos/archive/     Other or archived videos
data/               Training and annotation data
outputs/previews/   Generated preview images and videos
outputs/reports/    Generated Excel reports
docs/               Research and design documents
```

## Usage

```bash
venv/bin/python main.py --video videos/bat_vid2.mkv
```

On macOS, double-click `run_mac.command` to choose a video path interactively. On Windows, run `run_windows.bat` and enter the video path when prompted.

By default, the app loads both models:

- Bat model: `models/best3.pt`
- Wind turbine blade model: `models/blade_best.pt`

The blade model currently marks turbine or blade locations on the frame. It does not estimate collision risk by itself. To run bat detection only, add:

```bash
venv/bin/python main.py --video videos/bat_vid2.mkv --no-blade
```

If the blade model is stored elsewhere, specify its path:

```bash
venv/bin/python main.py --blade-model models/blade_best.pt
```

The app draws a red elliptical risk zone in the center of the frame. Once a confirmed bat track enters the zone, it is counted as a risk-zone bat and the percentage is shown below the count:

```text
Impact Risk: risk-zone bats / confirmed bats * 100%
```

This percentage is the share of detected bats that entered the risk zone. It is not a physical collision probability calibrated from real collision data.

If the turbine position differs between videos, adjust the ellipse with `--danger-zone`:

```bash
venv/bin/python main.py --danger-zone 400,240,160,100
```

The format is `centerX,centerY,radiusX,radiusY`. You can also include a rotation angle: `centerX,centerY,radiusX,radiusY,angle`.

## Playback Controls

- `Progress`: drag to a specific video frame. Seeking clears old tracks to avoid linking across unrelated segments.
- `Speed`: choose `0.25x`, `0.5x`, `1x`, `2x`, or `4x`.
- `Space`: pause or resume.
- `A` / `D`: jump backward or forward by 5 seconds.
- `[` / `]`: decrease or increase playback speed.
- `Q` or `Esc`: quit and export the Excel report.

`2x` and `4x` skip some frames and are useful for quickly locating segments. Use `1x` for full analysis.

Export a video:

```bash
venv/bin/python main.py --headless \
  --video-output outputs/previews/bat_result.mp4 \
  --excel-output outputs/reports/bat_result.xlsx
```

If very small targets are frequently missed, add `--detector tiled`. This is slower.

## Tests

```bash
venv/bin/python -m unittest discover -v
venv/bin/python main.py --headless --max-frames 3 --no-excel --tracker hybrid
venv/bin/python main.py --headless --max-frames 3 --no-excel --tracker motion
```

## Rebuild The Environment

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

Python 3.11 is recommended. The first install requires network access. The bat model should be placed at `models/best3.pt`, and the default blade model path is `models/blade_best.pt`.

## GitHub Collaboration

GitHub stores only source code and documentation. Large local files are excluded. After cloning, each collaborator should create their own environment:

```bash
git clone https://github.com/therealclvn/batdetector.git
cd batdetector
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

Models and videos should be shared separately and placed back at the expected paths:

```text
models/best3.pt
models/blade_best.pt
videos/bat_vid2.mkv
```

Excluded files include `venv/`, `models/*.pt`, `videos/*.mkv`, `videos/*.mp4`, `data/`, and `outputs/`.

## Share Package

The share package does not include large source videos, training data, or the virtual environment. After extracting it, run the launcher for your operating system and choose your own video file.
