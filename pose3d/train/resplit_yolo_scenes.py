"""Regroup an existing labeled synthetic dataset by complete BOP scene."""
from pathlib import Path
import argparse
import json
import random
from pose3d.train.build_composited_yolo_dataset import materialize, image_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-scenes', nargs='+', required=True)
    parser.add_argument('--val-scenes', nargs='+', required=True)
    parser.add_argument('--max-train-per-scene', type=int, default=0,
                        help='0 keeps all; otherwise retain a seeded sample per train scene.')
    parser.add_argument('--seed', type=int, default=20261005)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.out.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    train, val = set(args.train_scenes), set(args.val_scenes)
    if train & val:
        raise ValueError('scene sets overlap')
    counts = {'train': 0, 'val': 0, 'excluded': 0}
    if args.max_train_per_scene < 0:
        raise ValueError('max-train-per-scene cannot be negative')
    candidates = []
    for old_split in ('train', 'val', 'test'):
        directory = source / 'images' / old_split
        if directory.exists():
            candidates.extend(image_files(directory))
    selected_train = set()
    rng = random.Random(args.seed)
    for scene in sorted(train):
        scene_images = sorted(p.stem for p in candidates
                              if p.stem.startswith(f'synthetic_{scene}_'))
        n = args.max_train_per_scene or len(scene_images)
        selected_train.update(rng.sample(scene_images, min(n, len(scene_images))))
    seen = set()
    scene_counts = {}
    for old_split in ('train', 'val', 'test'):
        directory = source / 'images' / old_split
        if not directory.exists():
            continue
        for image in image_files(directory):
            if image.stem in seen:
                raise ValueError(f'duplicate ID: {image.stem}')
            seen.add(image.stem)
            if not image.stem.startswith('synthetic_'):
                raise ValueError(f'not a BOP synthetic image: {image}')
            scene = image.stem.split('_')[1]
            subset = 'train' if scene in train else 'val' if scene in val else None
            if subset == 'train' and image.stem not in selected_train:
                subset = None
            if subset is None:
                counts['excluded'] += 1
                continue
            label = source / 'labels' / old_split / (image.stem + '.txt')
            if not label.is_file():
                raise FileNotFoundError(label)
            materialize(image, output / 'images' / subset / image.name)
            materialize(label, output / 'labels' / subset / label.name)
            counts[subset] += 1
            scene_counts[scene] = scene_counts.get(scene, 0) + 1
    if not counts['train'] or not counts['val']:
        raise ValueError('empty train or val')
    for scene in train | val:
        if scene not in scene_counts:
            raise ValueError(f'scene not found: {scene}')
    (output / 'data.yaml').write_text(
        f'path: {output.as_posix()}\ntrain: images/train\nval: images/val\n'
        "nc: 1\nnames: ['action_camera']\n", encoding='utf-8')
    manifest = {'source': str(source), 'split_mode': 'scene',
                'train_scenes': sorted(train), 'val_scenes': sorted(val),
                'max_train_per_scene': args.max_train_per_scene, 'seed': args.seed,
                'counts': counts, 'scene_counts': scene_counts}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
