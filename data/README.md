# Data layout

FER2013 and RAF-DB are **not** used.

```
data/raw/scb/SCB5-Handrise-Read-write/.../images/{train,val}
data/raw/scb/SCB5-Handrise-Read-write/.../labels/{train,val}
data/raw/scb/.../scb.yaml
data/raw/daisee/                 ← DAiSEE clips (same download as SCB and L2CS)
third_party/SCB-dataset/
third_party/L2CS-Net/
third_party/yolov7/              ← loads official SCB weights
checkpoints/l2cs_gaze360_resnet50.safetensors
checkpoints/scb_yolo.pt
```

## Download everything

```powershell
python scripts/download_datasets.py
```

That downloads SCB, L2CS-Net, and DAiSEE. Add `--skip-daisee` only to leave out the ~14 GB clip set.

## Wired into the pipeline

| Asset | Config key | Used by |
|---|---|---|
| L2CS Gaze360 weights | `models.gaze_weights` | face gaze (L2CS-Net) |
| SCB YOLO weights | `models.scb_weights` | behavior boxes → HIGH/LOW |
| SCB images yaml | `data.scb_yaml` | `train_scb_yolo.py` |
| DAiSEE | `data.daisee_root` | fusion training |

## Sources

| Repo / dataset | Role |
|---|---|
| https://github.com/Whiffe/SCB-dataset.git | Student classroom behavior |
| https://github.com/Ahmednull/L2CS-Net.git | Gaze estimation |
| Hugging Face `wintonYF/SCB-Dataset` | SCB images + pretrained YOLO |
| Kaggle `olgaparfenova/daisee` | Optional engagement clips |
