# Third-party notices

MultiBlur is licensed under the GNU Affero General Public License v3.0 (see [LICENSE](LICENSE)).
It includes or uses the following third-party work, under their own licenses.

## License plate detection model

- **YOLOv11 license plate detector** by morsetechlab, `models/license-plate-finetune-v1s.mlpackage`
  (converted from https://huggingface.co/morsetechlab/yolov11-license-plate-detection).
  License: **AGPL-3.0**, the same as this project (see [LICENSE](LICENSE)).
- **Ultralytics** (https://github.com/ultralytics/ultralytics), used by the Python CLI and to export the model.
  License: **AGPL-3.0**.

## Face detection

- **CenterFace** model, `models/CenterFace.mlpackage` (converted from deface's `centerface.onnx`).
  Original project: https://github.com/Star-Clouds/CenterFace. License: **MIT**.
- **deface** (https://github.com/ORB-HD/deface): the macOS app ports its CenterFace decoding and
  non-maximum suppression to Swift; the Python CLI uses the package. License: **MIT**.

## Face recognition (editor)

- **SFace** from OpenCV Zoo, `models/SFace.mlpackage` (converted from
  https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface).
  License: **Apache License 2.0**, full text in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).
  The model was converted to CoreML; its weights were not otherwise modified.

---

### CenterFace — MIT License

```
MIT License

Copyright (c) 2019 StarClouds

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### deface — MIT License

```
MIT License

Copyright (c) 2020 Martin Drawitsch

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
