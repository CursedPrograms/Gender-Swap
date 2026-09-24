"""Pretrained weight locations and first-run download."""
import sys
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / 'pretrained'

FILES = {
    # e4e encoder + StyleGAN2 FFHQ 1024 generator in one checkpoint (Tov et al., 2021).
    'e4e': ('e4e_ffhq_encode.pt',
            'https://huggingface.co/AIRI-Institute/HairFastGAN/resolve/main/'
            'pretrained_models/encoder4editing/e4e_ffhq_encode.pt'),
    # Gender direction in StyleGAN2 FFHQ W+ space (18x512).
    'gender': ('gender.npy',
               'https://raw.githubusercontent.com/a312863063/generators-with-stylegan2/'
               'master/latent_directions/gender.npy'),
    # FairFace gender classifier (ViT, int8 ONNX) for choosing the swap direction.
    'gender_cls': ('fairface_gender_int8.onnx',
                   'https://huggingface.co/onnx-community/fairface_gender_image_detection-ONNX/'
                   'resolve/main/onnx/model_int8.onnx'),
    # OpenCV YuNet face detector with 5-point landmarks.
    'yunet': ('face_detection_yunet_2023mar.onnx',
              'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/'
              'models/face_detection_yunet/face_detection_yunet_2023mar.onnx'),
}


def _download(url, dest):
    tmp = dest.with_suffix(dest.suffix + '.part')
    print(f'Downloading {dest.name} ...')

    def report(blocks, block_size, total):
        if total > 0:
            done = min(blocks * block_size, total)
            sys.stdout.write(f'\r  {done / 2**20:7.1f} / {total / 2**20:.1f} MB')
            sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=report)
    print()
    tmp.replace(dest)


def ensure_all():
    WEIGHTS_DIR.mkdir(exist_ok=True)
    paths = {}
    for key, (name, url) in FILES.items():
        path = WEIGHTS_DIR / name
        if not path.exists():
            _download(url, path)
        paths[key] = path
    return paths
