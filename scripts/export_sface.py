"""Converts OpenCV's SFace face recognition model to CoreML for the macOS app's editor.

SFace (opencv_zoo, Apache 2.0) turns an aligned 112x112 face into a 128-number embedding: two views of
the same person give close embeddings. Input: download face_recognition_sface_2021dec.onnx from
https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface into models/.
Output: models/SFace.mlpackage (RGB image input, values 0-255 like OpenCV's FaceRecognizerSF).
"""
import os

import coremltools as ct
import numpy as np
import onnx
import torch
from onnx2torch import convert

ROOT = os.path.join(os.path.dirname(__file__), "..", "models")
model = convert(onnx.load(os.path.join(ROOT, "face_recognition_sface_2021dec.onnx"))).eval()
example = torch.rand(1, 3, 112, 112) * 255
traced = torch.jit.trace(model, example)

mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(name="image", shape=example.shape, color_layout=ct.colorlayout.RGB, scale=1.0)],
    outputs=[ct.TensorType(name="embedding")],
    minimum_deployment_target=ct.target.macOS14,
)
mlmodel.short_description = "SFace face recognition embedding (opencv_zoo, Apache 2.0)"
mlmodel.save(os.path.join(ROOT, "SFace.mlpackage"))
print("Saved", os.path.abspath(os.path.join(ROOT, "SFace.mlpackage")))
