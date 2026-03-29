title: Avian Vocal Classification System
emoji: 🦜
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860

# Avian Vocal Classification System

**Avian Vocal Classification System** is an end-to-end pipeline for **bird vocalization classification** using recordings from [Xeno-canto](https://xeno-canto.org/). The project targets **Indian bird species for data collection**, but the **trained model predicts vocalization type** (e.g. call, song, alarm call, flight call) using Xeno-canto metadata—not species identity.

Repository / package name: `avian-vocal-classification-system` (kebab-case).

Implemented pieces:

1. Download raw recordings per species (`src/download.py`)
2. Preprocess audio: resample, mono, trim silence, 5 s chunks (`src/preprocess.py`)
3. Augmented mel spectrograms saved as PNG (`src/features.py`)
4. Stratified train/val/test split by **recording ID** (no leakage) and label encoding (`src/dataset.py`)
5. **EfficientNet-B0** classifier (`src/model.py`), training with validation and best-checkpoint saving (`src/train.py`)
6. Evaluation: confusion matrix + classification report (`src/evaluate.py`)
7. Inference on raw audio: mel → image → model (`src/predict.py`)
8. **Django 6** web UI: upload audio, show predicted class + confidence (`webapp/`)

## How it fits together

- **`configs/config.yaml`** lists species for downloading and paths for data, splits, and model checkpoint.
- **`BirdSoundDataset`** scans spectrogram PNGs, builds stratified splits from CSV metadata, copies files into `SPLIT_DIR`, and writes `models/split_index.json` plus `models/class_mapping.json`.
- **`BirdSplitDataset`** loads images from the saved index for training/evaluation.
- The **web app** loads `models/checkpoints/best_model.pth` and `models/class_mapping.json` (paths are set in `webapp/views.py`).

## Project structure

```text
avian-vocal-classification-system/
├── config/                    # Django project (settings, root urls)
├── webapp/                    # Upload + predict UI (templates, views)
├── manage.py                  # Django entrypoint
├── configs/
│   └── config.yaml            # Species list + directory paths + model paths
├── data/                      # raw MP3/CSV, processed WAV, spectrograms, split copies (typical layout)
├── models/
│   ├── checkpoints/           # best_model.pth (from training)
│   ├── class_mapping.json     # int id → label name (written by dataset step)
│   └── split_index.json       # train/val/test paths + labels (written by dataset step)
├── outputs/                   # confusion matrix and other evaluation outputs
├── logs/                      # per-module logs
├── src/
│   ├── __init__.py            # Orchestration: data stages + train (see below)
│   ├── download.py
│   ├── preprocess.py
│   ├── features.py
│   ├── dataset.py             # BirdSoundDataset, BirdSplitDataset
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── logger.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Requirements

- Python 3.12
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Xeno-canto API key for downloading

## Setup

### 1) Install dependencies

From the repository root:

```bash
uv sync
```

### 2) Environment

Create `.env` in the project root:

```env
XENO_CANTO_API_KEY=your_api_key_here
```

`src/download.py` loads this with `python-dotenv`.

### 3) Configure species and paths

Edit `configs/config.yaml`: set `species_list` and paths such as `RAW_DIR`, `PROCESSED_DIR`, `SPECTROGRAM_DIR`, `SPLIT_DIR`, `MODEL_PATH`, and `CONFUSION_MATRIX_PATH`. Paths are written relative to how you run scripts (see below).

Example species entries:

```yaml
species_list:
  - scientific_name: "Pycnonotus cafer"
    common_name: "Red-vented Bulbul"
  # ...
```

## Running the pipeline

Scripts under `src/` use imports and paths that assume you run from **`src/`** (as in the original layout), so:

```bash
cd src
uv run python __init__.py
```

By default, `src/__init__.py` runs **training** (train/val loaders, model build, `Train.train()`). Other stages are commented out at the bottom of the file—uncomment or call them as needed.

Typical order of operations:

1. **`download()`** — fetch MP3s and per-species CSV metadata into `RAW_DIR`.
2. **`preprocess()`** — WAV chunks in `PROCESSED_DIR`.
3. **`feature_extraction()`** — mel spectrogram PNGs under `SPECTROGRAM_DIR`.
4. **Build split index** — instantiate `BirdSoundDataset` with your `SPLIT_DIR`, `RAW_DIR`, and `SPECTROGRAM_DIR`, then call `build_and_save_index("../models/split_index.json")` (adjust path if your cwd differs). This also writes `models/class_mapping.json`.
5. **Train** — run `__main__` in `src/__init__.py` (or wire the same calls) so `dataset('train'|'val'|'test')` uses `models/split_index.json` and saves the best weights to `MODEL_PATH`.
6. **`evaluate(...)`** — uncomment the evaluate call in `__init__.py` after training to run the test loader and write metrics/plots per `CONFUSION_MATRIX_PATH`.

## Model and features

- **Backbone:** `torchvision` EfficientNet-B0 (ImageNet weights), custom classifier head for `num_classes`.
- **Input:** PNG spectrograms, 224×224, ImageNet normalization (same in `BirdSplitDataset` and `Predictor`).
- **Mel params (feature extraction):** e.g. `n_mels=128`, `fmin=500`, `fmax=8000`, `hop_length=512`, sample rate 22050 Hz. Prediction pads/trims audio to 5 s and matches this mel setup.

## Web app (Django)

From the **repository root** (where `manage.py` lives):

```bash
uv run python manage.py runserver
```

Open the site root URL; upload an audio file. The app uses `Predictor` with `models/checkpoints/best_model.pth` and `models/class_mapping.json` (see `webapp/views.py` if you need to change paths).

For production, configure `SECRET_KEY`, `DEBUG`, `ALLOWED_HOSTS`, and static/media settings appropriately.

## Logging

Logs go under `logs/` with a shared format (`download`, `preprocess`, `features`, `train`, `evaluate`, `predict`, pipeline modules).

## Troubleshooting

- **`XENO_CANTO_API_KEY is not set`** — ensure `.env` exists at the project root and the variable name matches what `download.py` expects.
- **`split_index.json` / empty loaders** — run the dataset builder after spectrograms exist; confirm paths in `config.yaml` and that `models/split_index.json` was generated.
- **`ModuleNotFoundError` for `src` in Django** — `webapp/views.py` adds `src` to `sys.path`; run `manage.py` from the repo root.
- **Path confusion** — prefer running pipeline commands from `src/` or refactor paths to `Path(__file__).resolve()` if you want to run from the repo root only.

## Optional extensions

- Tighten config-driven hyperparameters in `configs/config.yaml`.
- Add API tests and CI; harden the Django deployment settings.
- Species-level classification would require a different label source and training target than the current vocalization-type setup.
