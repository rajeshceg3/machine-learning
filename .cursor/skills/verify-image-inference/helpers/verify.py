#!/usr/bin/env python3
"""Drive the real three image CLIs, keeping all output outside source."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
p = argparse.ArgumentParser()
p.add_argument('mode', choices=['doctor', 'drive'])
p.add_argument('out', type=Path)
a = p.parse_args()
out = a.out.resolve()
out.mkdir(parents=True, exist_ok=True)
result = {'mode': a.mode, 'ok': False, 'checks': []}
try:
    for source in ROOT.glob('*.py'):
        ast.parse(source.read_text(), filename=str(source))
    for directory in ['faceDetectionModel_ssd', 'imgClassifierModel_googlenet', 'imgClassifierModel_mobilenet']:
        weights = next((ROOT / directory).glob('*.caffemodel'))
        assert weights.stat().st_size > 1_000_000
        result['checks'].append({'model': str(weights.relative_to(ROOT)), 'sha256': hashlib.sha256(weights.read_bytes()).hexdigest()})
    result['runtime'] = {'opencv': cv2.__version__, 'numpy': np.__version__}
    if a.mode == 'drive':
        scripts = [('faceDetectionOpenCV_dnn', 'Lenna.png', 'face'), ('imageClassifierOpenCV_dnn_googlenet', 'eagle.jpg', 'bald eagle'), ('imageClassifierOpenCV_dnn_mobilenet', 'bookshelf_cat.jpg', 'cat')]
        with tempfile.TemporaryDirectory(prefix='image-inference-') as scratch:
            for script, image, expected in scripts:
                target = out / (script + '.png')
                details = out / (script + '.json')
                cmd = [sys.executable, str(ROOT / (script + '.py')), '--output', str(target), '--json', str(details)]
                q = subprocess.run(cmd, cwd=scratch, capture_output=True, text=True, timeout=60)
                (out / (script + '.log')).write_text(q.stdout + q.stderr)
                assert q.returncode == 0, q.stderr
                data = json.loads(details.read_text())['predictions']
                assert data and data[0]['label'] == expected
                assert all(0 <= x['probability'] <= 1 for x in data)
                annotated = cv2.imread(str(target))
                original = cv2.imread(str(ROOT / image))
                assert annotated.shape == original.shape and np.any(annotated != original)
                for item in data:
                    if 'box' in item:
                        x1,y1,x2,y2 = item['box']
                        assert 0 <= x1 < x2 <= original.shape[1] and 0 <= y1 < y2 <= original.shape[0]
                missing_output = out / (script + '-must-not-exist.png')
                assert not missing_output.exists(), 'Use a fresh output directory'
                bad = subprocess.run([sys.executable, str(ROOT / (script + '.py')), '--image', str(Path(scratch) / 'missing.png'), '--output', str(missing_output)], cwd=scratch, capture_output=True, text=True, timeout=60)
                assert bad.returncode != 0 and 'Image could not be read' in bad.stderr and not missing_output.exists()
                result['checks'].append({'cli': script, 'from_unrelated_cwd': True, 'prediction': data, 'annotation_changed_pixels': True, 'missing_image_fails_without_output': True})
    result['ok'] = True
except Exception as exc:
    result['failure'] = str(exc)
finally:
    (out / (a.mode + '.json')).write_text(json.dumps(result, indent=2))
print(json.dumps(result))
sys.exit(0 if result['ok'] else 1)
