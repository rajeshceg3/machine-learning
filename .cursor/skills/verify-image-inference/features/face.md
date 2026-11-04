# Still-image face detection

## Sub-features
Runs supplied SSD Caffe model on Lenna.png; inspect red box around the visible face and probability text. No identity inference.

## How to get to it (user POV)
Source entry: faceDetectionOpenCV_dnn.py. Run from the command line, not a web UI.

## Driving it with Python CLI
Use doctor/drive in SKILL.md for real subprocess inference and missing-input checks.

## Gotchas
Headless output is the proved route, not GUI/device or training accuracy. Preserve evidence and bundled data.
