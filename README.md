# OCR Calendar Digitization

A deep learning pipeline for digitizing handwritten paper calendars using semantic segmentation and image classification models. The system detects and parses calendar regions, then reads day numbers, month names, and handwritten text annotations from scanned calendar images.

---

## Overview

The project tackles the problem of extracting structured data from scanned handwritten calendars. It does so through a **multi-stage pipeline**:

1. **Calendar Parsing** — a U-Net segmentation model labels each pixel of a calendar image into one of six semantic classes: `none`, `month`, `day`, `note`, `emptyNote`, `border`.
2. **Day Digitization** — a ResNet-6 classifier recognizes the day number (1–31) from cropped day-cell images.
3. **Month Digitization** — a ResNet-6 classifier recognizes the month name from cropped month-cell images.
4. **Text/Word Recognition** — a Transformer-based model (trained on the IAM Handwriting dataset) reads handwritten text annotations inside calendar note fields.

All models are implemented with [PyTorch Lightning](https://lightning.ai/) and experiment tracking is handled via [Weights & Biases](https://wandb.ai/).

---

## Repository Structure

```
OCR-calendar_digitization/
├── configs/                        # YAML configuration files (one per model)
│   ├── config_parserCalendar.yaml  # Calendar segmentation (U-Net)
│   ├── config_digitDay.yaml        # Day digit classifier (ResNet-6)
│   ├── config_digitMonth.yaml      # Month classifier (ResNet-6)
│   ├── config_digitText.yaml       # Text recognition (Transformer)
│   └── config_parserWord.yaml      # Word/line segmentation (U-Net)
│
├── scripts/                        # Training entry points
│   ├── train_parserCalendar.py
│   ├── train_digitDay.py
│   ├── train_digitMonth.py
│   ├── train_digitText.py
│   └── train_parserWord.py
│
├── src/
│   ├── model/                      # Model definitions
│   │   ├── calendar_parser.py      # U-Net wrapper — calendar segmentation
│   │   ├── day_digitizer.py        # ResNet-6 wrapper — day classification
│   │   ├── month_digitizer.py      # ResNet-6 wrapper — month classification
│   │   ├── word_parser.py          # U-Net wrapper — word segmentation
│   │   ├── word_digitalizer.py     # Transformer wrapper — text recognition
│   │   ├── resnet6.py              # Custom 6-layer ResNet backbone
│   │   └── unet.py                 # U-Net architecture
│   │
│   ├── data/                       # Dataset and DataModule classes
│   │   ├── calendardb.py           # CalendarDataset (segmentation)
│   │   ├── calendarloader.py       # CalendarDataModule
│   │   ├── daydb.py                # NumberDataset for day cells
│   │   ├── dayloader.py            # Day DataModule
│   │   ├── monthdb.py              # Dataset for month cells
│   │   ├── monthloader.py          # Month DataModule
│   │   ├── iamdb.py                # IAM Handwriting Dataset wrapper
│   │   └── iamloader.py            # IAM DataModule
│   │
│   └── utils/
│       ├── utility.py              # General image utilities
│       ├── utility_cv_pyr.py       # Image pyramid / OpenCV helpers
│       ├── dice_loss.py            # Soft Dice Loss
│       ├── plotting.py             # Confusion matrix plotting
│       └── annotation.py          # VGG annotation reader
│
├── model_data/                     # Pre-built label dictionaries (pickle)
│   ├── digit_dictionary.dict       # Day digits vocabulary
│   ├── month_dictionary.dict       # Month names vocabulary
│   └── segment_dictionary.dict     # Segmentation class map
│
├── data/                           # Raw dataset (not tracked in git)
├── data_split/                     # Train/val split files
├── ckpt/                           # Model checkpoints
├── cm/                             # Saved confusion matrix images
├── environment.yml                 # Conda environment specification
└── pyproject.toml                  # Package build configuration
```

---

## Models

### CalendarParser (U-Net)
Semantic segmentation model that labels each pixel of a full calendar scan. Trained with **Soft Dice Loss** and evaluated with micro-averaged **F1 score**.

| Class | Description |
|---|---|
| `none` | Background / contour |
| `month` | Month header cell |
| `day` | Numbered day cell |
| `note` | Handwritten note area |
| `emptyNote` | Empty note area |
| `border` | Calendar border |

### DayDigitalizer / MonthDigitalizer (ResNet-6)
Lightweight custom ResNet with 6 convolutional layers and two residual blocks, designed to classify small cropped images. Trained with **CrossEntropy Loss**, preprocessed via `ViTImageProcessor` (32×32 resize, normalization). Data augmentation includes Gaussian noise, dropout, blur, and translation.

### WordParser (U-Net)
Binary segmentation model that isolates handwritten word regions from note cells (classes: `none`, `word`).

### Text Recognizer (Transformer)
Sequence-to-sequence model fine-tuned on the [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) for reading handwritten words and lines.

---

## Installation

### Prerequisites
- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.12
- A CUDA-compatible GPU, Apple Silicon (MPS), or CPU

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/OCR-calendar_digitization.git
cd OCR-calendar_digitization

# Create and activate the conda environment
conda env create -f environment.yml
conda activate ocr

# Install the project package in editable mode
pip install -e .
```

---

## Data Preparation

Place your datasets under the `data/` directory following this structure:

```
data/
├── calendars/              # Full calendar scans (.png or .jpg)
│   ├── calendars/          # Raw calendar images
│   └── calendars_masks/    # VGG-annotated JSON ground truth masks
├── days/                   # Cropped day-cell images
├── months/                 # Cropped month-cell images
└── IAM_handwriting/        # IAM dataset (words / lines splits)
```

The `data_split/` folder will be auto-populated on first run when `reload_data: False` is set in the config.

---

## Training

Each model is trained via its corresponding script. Configuration is loaded from YAML files in `configs/`.

```bash
# 1. Train the calendar segmentation model
python scripts/train_parserCalendar.py

# 2. Train the day digit classifier
python scripts/train_digitDay.py

# 3. Train the month classifier
python scripts/train_digitMonth.py

# 4. Train the text recognizer (words)
python scripts/train_digitText.py

# 5. Train the word/line segmentation model
python scripts/train_parserWord.py
```

### Configuration

Edit the relevant YAML file in `configs/` before training.


## Dependencies

Key packages (see `environment.yml` for full list):

| Package | Purpose |
|---|---|
| `torch` + `lightning` | Model training framework |
| `transformers` + `tokenizers` | Transformer-based text recognition |
| `torchmetrics` | F1 score, confusion matrix metrics |
| `scikit-image` + `opencv-python` | Image preprocessing |
| `imgaug` | Data augmentation |
| `wandb` | Experiment tracking |
| `numpy==1.26.4` | Numerical computing |

---
