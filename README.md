# machine-learning

Three still-image demos now support `--image PATH`, `--output annotated.png` and `--json predictions.json`. With `--output`, inference is headless. Without it, they retain the OpenCV GUI workflow, which needs GUI OpenCV and a display.

For a pinned isolated runtime and actual real-model proof, follow [.cursor/skills/verify-image-inference/SKILL.md](.cursor/skills/verify-image-inference/SKILL.md). This covers supplied face/eagle/cat images and missing-image failures, not model accuracy or the entire repository. Video/tracker, C++ and Keras examples remain separate unverified surfaces with known prerequisites/defects.
