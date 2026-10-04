# Classroom multimodal attention pipeline

Six-stage system. **FER2013 and RAF-DB are not used.**

Gaze comes from **L2CS-Net**. Classroom actions come from **SCB-dataset**.

1. **Source** — classroom video → frames
2. **Detect & track** — YOLO11-Pose + DeepSORT
3. **Modalities** — L2CS-Net gaze + SCB behavior (hand-raising / reading / writing / …)
4. **Attention model** — rubric + optional fusion score 0–100
5. **Trigger** — attention-drop detector
6. **Intervention** — teacher recommendation

## Install

```powershell
cd "C:\Users\USER\Desktop\Thesis mmodel"
python -m pip install -e .
python -m pip install ultralytics deep-sort-realtime safetensors huggingface_hub kagglehub
```

## Download SCB-dataset, L2CS-Net, and DAiSEE

```powershell
python scripts/download_datasets.py
```

Or the lower-level script, which downloads the same three:

```powershell
python scripts/setup_scb_l2cs.py
```

That clones:

- https://github.com/Whiffe/SCB-dataset.git → `third_party/SCB-dataset`
- https://github.com/Ahmednull/L2CS-Net.git → `third_party/L2CS-Net`
- YOLOv7 (needed to run the official SCB weights) → `third_party/yolov7`

and downloads:

- L2CS Gaze360 weights → `checkpoints/l2cs_gaze360_resnet50.safetensors`
- SCB pretrained YOLO (hand-raising / reading / writing) → `checkpoints/scb_yolo.pt`
- SCB images/labels → `data/raw/scb/`
- DAiSEE clips → `data/raw/daisee/` (Kaggle `olgaparfenova/daisee`, about 14 GB)

DAiSEE needs a Kaggle login the first time:

```powershell
python -c "import kagglehub; kagglehub.login()"
```

Add `--skip-daisee` only if you want SCB and L2CS without the 14 GB clip set.

After this, the classroom overlay uses SCB: **reading** and **writing** count as HIGH attention. Retraining is optional:

```powershell
python scripts/train_scb_yolo.py
```

## Train on SCB, L2CS-Net, and DAiSEE

```powershell
python scripts/train_all.py
```

That uses every SCB image (train and val) and every labeled DAiSEE clip (train, validation, and test). L2CS-Net is fine-tuned on a face from each of those clips. A checkpoint is saved every 30 minutes. Run the same command again to continue after a power cut.

## Smoke / benchmark test (before full training)

```powershell
python scripts/benchmark_smoke.py
```

Uses **40 SCB images**, **20 DAiSEE clips**, **40 L2CS face samples**, then prints precision / recall / F1 / accuracy and writes `runs/smoke_metrics.json`. Classroom overlay: `runs/attention_overlay_smoke.mp4`.

## Smoke test (before full training)

Trains on a **tiny** subset, then overlays your classroom video:

- 20 SCB images → mini YOLO dry-run (`checkpoints/scb_yolo_smoke.pt`)
- 10 DAiSEE clips → mini fusion dry-run (`checkpoints/fusion_smoke.pt`)
- Classroom test uses pretrained SCB + L2CS + rubric (the 20-image YOLO is too small to detect well)

```powershell
python scripts/smoke_test_train.py
```

Output: `runs/attention_overlay_smoke.mp4`

## Run your classroom video

```powershell
python -m attention_pipeline.cli "video\Classroom_video.mp4" --seconds 20
```

Open `runs/attention_overlay.mp4`.

## What is trained vs frozen

| Piece | Source | Train? |
|---|---|---|
| YOLO11-Pose | Ultralytics | Frozen |
| L2CS-Net gaze | Ahmednull/L2CS-Net Gaze360 weights | Pretrained (optional fine-tune) |
| SCB behavior YOLO | Whiffe/SCB-dataset pretrained `scb_yolo.pt` | Optional `train_scb_yolo.py` |
| DAiSEE | Kaggle `olgaparfenova/daisee` | Optional fusion train |
| FER2013 / RAF-DB | — | **Not used** |

SCB-dataset3 classes: hand-raising, reading, writing, using-phone, bowing-head, leaning-over-table. The Hugging Face zip currently used is the 3-class subset (hand-raising, reading, writing).
