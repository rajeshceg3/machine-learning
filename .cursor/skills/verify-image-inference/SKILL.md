---
name: verify-image-inference
description: Run machine-learning's actual three still-image Caffe inference CLIs, inspect annotations, and prove missing-image failures after model or example changes.
---
# Image inference verification

## Launch
From the repository root create a private runtime: `python3 -m venv /tmp/ml-verify-runtime`, then `/tmp/ml-verify-runtime/bin/pip install -r .cursor/skills/verify-image-inference/requirements.txt`. No server or credentials. Bundled real Caffe weights and sample images are required; no models are fetched at runtime. Headless requirements intentionally do not supply GUI support.
Run a real example: `/tmp/ml-verify-runtime/bin/python faceDetectionOpenCV_dnn.py --output /tmp/ml-face.png --json /tmp/ml-face.json`. The two imageClassifierOpenCV_dnn scripts accept the same options. `--image PATH` overrides the sample; paths otherwise resolve beside source, independent of current directory. Without --output the original interactive OpenCV window behavior remains, but needs a GUI OpenCV build/display and has not been proved in the headless runtime.

## Doctor
`/tmp/ml-verify-runtime/bin/python .cursor/skills/verify-image-inference/helpers/verify.py doctor /tmp/ml-proof` parses Python files, identifies installed versions and hashes three actual model weights. This is not inference coverage. After a failed drive rerun doctor then a fresh drive; no persistent instance exists.

## Drive
`/tmp/ml-verify-runtime/bin/python .cursor/skills/verify-image-inference/helpers/verify.py drive /tmp/ml-proof` launches all three real scripts from an unrelated temporary directory. Each must write an annotated PNG and JSON with bounded probabilities and correct sample label, image dimensions and changed pixels. Face/cat boxes must be in bounds. Each script must reject a missing-image input with a nonzero exit and no output. Expected supplied samples: face on Lenna, bald eagle on eagle, cat on bookshelf image. Model confidence is not real-world accuracy; cat confidence is only about0.416 in this runtime.

## Evidence
Named output has doctor.json, drive.json, each script's PNG/JSON/stdoutstderr log and model hashes. Inspect actual annotations, not just changed pixel count or labels. These are real forward passes, not mocks or test-only inference endpoints. No recognition of a person's identity is performed. Changing weights/sample data requires reviewing new output, not loosening expected assertions to get green.

## Cleanup
Helper runs bounded subprocesses and removes only its own temporary working directory after every attempt. No shared GUI/camera/server is touched. Evidence survives outside temporary work. A runtime may be reused, but use distinct output directories for separate runs and confirm files after cleanup. Never remove bundled datasets/models or proof.

## Coverage and maintenance
Only three still-image Python workflows are executed. VideoFace/videoClassifier, dlib trackers, C++ examples and Keras notebooks are mapped but remain unverified prerequisites, not passing features. Video/tracker source has model path/EOF and single-tracker API defects still owned for separate product repair; do not present this as full repository health. GUI default, new arbitrary images, model quality, GPU, datasets/training and deployment are unverified. Use /maintain-verification-skill as examples evolve; product fixes belong separately from map-only maintenance.
