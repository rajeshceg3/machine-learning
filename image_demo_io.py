"""Shared CLI I/O for the three still-image demos; GUI remains the default."""
import argparse
import json
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent


def options(default_image):
    parser = argparse.ArgumentParser(description='Run bundled Caffe inference on a local image.')
    parser.add_argument('--image', type=Path, default=ROOT / default_image)
    parser.add_argument('--output', type=Path, help='Save annotated image instead of opening a GUI.')
    parser.add_argument('--json', type=Path, help='Write prediction details as JSON.')
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f'Image could not be read: {args.image}')
    return args, image


def model(directory, prototxt, weights):
    return cv2.dnn.readNetFromCaffe(str(ROOT / directory / prototxt), str(ROOT / directory / weights))


def finish(args, image, predictions):
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), image):
            raise OSError(f'Image could not be written: {args.output}')
    else:
        cv2.imshow('Prediction', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({'input': str(args.image), 'predictions': predictions}, indent=2))
    print(json.dumps({'predictions': predictions}))
