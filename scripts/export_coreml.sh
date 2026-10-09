#!/bin/bash
# Converts the plate (YOLO) and face (deface's CenterFace) models to CoreML for the macOS app.
# Uses a dedicated environment: coremltools needs torch 2.7 (the main venv has a newer torch).
set -euo pipefail
cd "$(dirname "$0")/.."

SIZE=${1:-s}
ENV=.venv-export

if [ ! -x "$ENV/bin/python" ]; then
    uv venv -q --python 3.12 "$ENV"
    uv pip install -q --python "$ENV/bin/python" "torch==2.7.0" "torchvision==0.22.0" "coremltools==9.0" "numpy<2.3" ultralytics onnx onnx2torch deface
fi
[ -f "models/license-plate-finetune-v1$SIZE.pt" ] || .venv/bin/python multiblur.py --download "$SIZE"

YOLO_OFFLINE=1 "$ENV/bin/python" -c "
from ultralytics import YOLO
YOLO('models/license-plate-finetune-v1$SIZE.pt').export(format='coreml', nms=True, imgsz=640)"

"$ENV/bin/python" scripts/export_centerface.py
