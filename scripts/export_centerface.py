"""Converts deface's CenterFace ONNX model to CoreML for the macOS app.

Output: models/CenterFace.mlpackage with a flexible RGB image input (multiples of 32,
up to 2048 px per side, like deface which runs at full resolution) and the four raw
CenterFace outputs (heatmap, scale, offset, landmarks); decoding happens in Swift.
"""
import os

import coremltools as ct
import deface
import onnx
import torch
from onnx2torch import convert

ONNX_PATH = os.path.join(os.path.dirname(deface.__file__), "centerface.onnx")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "CenterFace.mlpackage")
SIZE = 640
MAX_SIZE = 2048

model = convert(onnx.load(ONNX_PATH)).eval()
example = torch.rand(1, 3, SIZE, SIZE) * 255
traced = torch.jit.trace(model, example)

mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(
        name="image",
        shape=ct.Shape((1, 3, ct.RangeDim(32, MAX_SIZE, default=SIZE), ct.RangeDim(32, MAX_SIZE, default=SIZE))),
        color_layout=ct.colorlayout.RGB,
        scale=1.0,
    )],
    outputs=[ct.TensorType(name=n) for n in ("heatmap", "scale", "offset", "landmarks")],
    minimum_deployment_target=ct.target.macOS14,
)
mlmodel.short_description = "CenterFace face detector (from deface, MIT license)"
mlmodel.save(OUT_PATH)
print(f"Saved {os.path.abspath(OUT_PATH)}")
