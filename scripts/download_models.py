"""Download published DDMF models and verify their SHA-256 checksums."""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

REPO = 'https://huggingface.co/AlexLiu2023/DDMF/resolve/main/'
ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'tooth': ('tooth_iter_160000.pth', '71579ab5eadd730bbaae0e2eb773a8ccc978869cc19a35bc939117ec08325e3d',
              'code/cbct-segmentation/work_dirs/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K_best_1:2/iter_160000.pth'),
    'bone': ('bone_iter_160000.pth', '6868d4ea7ad1c0dfd269696493e1c82afa9e2b7784e112d15abc16f056d523be',
             'code/jaw-segmentation/work_dirs/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K/iter_160000.pth'),
    'ios': ('ios_model.onnx', 'ee7c8447dba3c74a2a11daa208c37806b6e221bb2d41da29edf5353aee364557',
            'code/seg_iosnet/deployment/app/model.onnx'),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(name, root):
    filename, expected, relative = MODELS[name]
    target = root / relative
    if target.exists():
        if sha256(target) != expected:
            raise ValueError('Existing file has a different checksum; not overwriting: ' + str(target))
        print('Verified existing ' + str(target))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.part', delete=False) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(REPO + filename, timeout=120) as response:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
        if sha256(temporary) != expected:
            raise ValueError('Checksum mismatch for ' + filename)
        # Exclusive creation protects a file created by another downloader.
        os.link(temporary, target)
        print('Downloaded and verified ' + str(target))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['all'] + list(MODELS), default='all')
    parser.add_argument('--root', type=Path, default=ROOT, help='DDMF repository root')
    args = parser.parse_args()
    for name in MODELS if args.model == 'all' else [args.model]:
        download(name, args.root)


if __name__ == '__main__':
    main()
