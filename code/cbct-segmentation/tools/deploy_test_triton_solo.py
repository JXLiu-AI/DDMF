import argparse
import os
import tqdm
import numpy as np
import tritoninferencer as tif

from PIL import Image


"""
input: 'input'
output: 'output'
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='mmseg backend test (and eval)')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--url', required=True)
    parser.add_argument('--show-dir', help='directory where painted images will be saved')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.show_dir, exist_ok=True)

    sess = tif.create_client(args.url)

    for name in tqdm.tqdm(os.listdir(args.input_dir)):
        if not name.endswith('.jpg'):
            continue

        img = Image.open(os.path.join(args.input_dir, name))

        origin_size = img.size

        img = np.asarray(img.resize((512, 512)))
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        img = (img - np.asarray([[[123.675, 116.28, 103.53]]])) / np.asarray([[[58.395, 57.12, 57.375]]])
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)[np.newaxis]

        seg_pred = tif.infer('cbct-segmenter', {'input': img}, sess)['output'][0][0]

        seg_pred = np.asarray(seg_pred*255, dtype=np.uint8)
        seg_pred = Image.fromarray(seg_pred).resize(origin_size, Image.NEAREST)
        seg_pred.save(os.path.join(args.show_dir, name.replace('.jpg', '.png')))


if __name__ == '__main__':
    main()
