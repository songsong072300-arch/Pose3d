"""Validate and append labeled real frames to YOLO train only, idempotently."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset.resolve()
    if not (root / 'data.yaml').is_file():
        raise FileNotFoundError(root / 'data.yaml')
    images = sorted(p for p in args.images.resolve().iterdir()
                    if p.suffix.lower() in {'.jpg', '.jpeg', '.png'} and not p.name.startswith('._'))
    if not images:
        raise ValueError('No source images')
    held_out = {digest(p) for subset in ('val', 'test')
                for p in (root / 'images' / subset).glob('*') if p.is_file()}
    plan = []
    for source in images:
        label = args.labels.resolve() / (source.stem + '.txt')
        text = label.read_text(encoding='utf-8-sig')
        boxes = []
        for line in text.splitlines():
            if not line.strip():
                continue
            values = np.asarray([float(v) for v in line.split()])
            if values.shape != (5,) or not np.isfinite(values).all() or values[0] != 0:
                raise ValueError(f'Invalid class/box: {label}: {line}')
            _, x, y, w, h = values
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError(f'Invalid normalized box: {label}: {line}')
            if min(x-w/2, y-h/2) < -1e-5 or max(x+w/2, y+h/2) > 1+1e-5:
                raise ValueError(f'Box extends outside image: {label}: {line}')
            boxes.append(values.tolist())
        image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Unreadable image: {source}')
        sha = digest(source)
        if sha in held_out:
            raise ValueError(f'Image also exists in val/test: {source}')
        name = f'real_extracted_{source.stem}'
        dest = root / 'images/train' / (name + source.suffix.lower())
        dest_label = root / 'labels/train' / (name + '.txt')
        if dest.exists() and digest(dest) != sha:
            raise FileExistsError(dest)
        if dest_label.exists() and dest_label.read_text(encoding='utf-8') != text:
            raise FileExistsError(dest_label)
        plan.append((source, dest, dest_label, text, boxes, sha))
    # Complete validation before making any dataset changes.
    audit = root / 'real_data_audit'
    audit.mkdir(exist_ok=True)
    records = []
    added = 0
    for index, (source, dest, label, text, boxes, sha) in enumerate(plan):
        if not dest.exists():
            shutil.copy2(source, dest)
            added += 1
        label.write_text(text, encoding='utf-8')
        records.append({'source': str(source), 'image': str(dest.relative_to(root)),
                        'label': str(label.relative_to(root)), 'sha256': sha, 'boxes': len(boxes)})
        if index in (0, len(plan)//2, len(plan)-1):
            image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
            height, width = image.shape[:2]
            for _, x, y, w, h in boxes:
                lo = (round((x-w/2)*width), round((y-h/2)*height))
                hi = (round((x+w/2)*width), round((y+h/2)*height))
                cv2.rectangle(image, lo, hi, (0, 220, 0), 3)
            cv2.imencode('.jpg', image)[1].tofile(audit / f'{source.stem}_boxes.jpg')
    report = {'source_images': str(args.images.resolve()), 'source_labels': str(args.labels.resolve()),
              'subset': 'train', 'images': len(plan), 'boxes': sum(len(p[4]) for p in plan),
              'records': records,
              'note': 'These real frames are training data and must not be used as independent final test.'}
    (root / 'real_data_manifest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'new_images_added': added, 'real_images': len(plan), 'real_boxes': report['boxes'],
                      'total_train_images': len(list((root/'images/train').glob('*.jpg'))),
                      'total_val_images': len(list((root/'images/val').glob('*.jpg')))}, indent=2))


if __name__ == '__main__':
    main()
