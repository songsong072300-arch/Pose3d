"""Create a deterministic visual audit sheet, without changing source images."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=24)
    parser.add_argument('--columns', type=int, default=4)
    parser.add_argument('--tile-width', type=int, default=320)
    parser.add_argument('--labels', type=Path,
                        help='Optional YOLO label directory; draw boxes for visual audit.')
    args = parser.parse_args()
    paths = sorted(p for p in args.images.iterdir()
                   if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if not paths:
        raise ValueError('no images')
    count = min(args.count, len(paths))
    indices = [round(i * (len(paths) - 1) / max(count - 1, 1)) for i in range(count)]
    height = args.tile_width * 3 // 4
    rows = (count + args.columns - 1) // args.columns
    sheet = Image.new('RGB', (args.columns * args.tile_width, rows * (height + 24)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, index in enumerate(indices):
        path = paths[index]
        with Image.open(path) as source:
            source = source.convert('RGB')
            if args.labels:
                label_path = args.labels / f'{path.stem}.txt'
                if not label_path.is_file():
                    raise FileNotFoundError(label_path)
                box_draw = ImageDraw.Draw(source)
                for line in label_path.read_text().splitlines():
                    category, cx, cy, width, box_height = map(float, line.split())
                    if not (0 < width <= 1 and 0 < box_height <= 1):
                        raise ValueError(f'invalid box size: {label_path}')
                    box = ((cx - width / 2) * source.width,
                           (cy - box_height / 2) * source.height,
                           (cx + width / 2) * source.width,
                           (cy + box_height / 2) * source.height)
                    box_draw.rectangle(box, outline='lime', width=3)
            tile = ImageOps.contain(source, (args.tile_width, height))
        x = i % args.columns * args.tile_width
        y = i // args.columns * (height + 24)
        sheet.paste(tile, (x, y))
        draw.text((x + 4, y + height + 4), path.name, fill='black')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f'{count} of {len(paths)} source images: {args.out.resolve()}')


if __name__ == '__main__':
    main()
