# Still-image eagle and cat

## Sub-features
Real supplied GoogLeNet outputs bald eagle, and MobileNetSSD outputs cat on bookshelf_cat.jpg. Inspect label/annotation and bounded box. Cat confidence is low, not an accuracy guarantee.

## How to get to it (user POV)
Source entry: imageClassifierOpenCV_dnn_googlenet.py and imageClassifierOpenCV_dnn_mobilenet.py. Run from the command line, not a web UI.

## Driving it with Python CLI
Use doctor/drive in SKILL.md for real subprocess inference and missing-input checks.

## Gotchas
Headless output is the proved route, not GUI/device or training accuracy. Preserve evidence and bundled data.
