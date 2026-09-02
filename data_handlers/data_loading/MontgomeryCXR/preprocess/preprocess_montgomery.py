"""Preprocess the raw Montgomery County CXR set into this repo's standard
per-sample-folder layout (same as Camus / CelebAMask-HQ).

Raw (see ../montgomery_explore.ipynb for the look at it):
  data/MontgomeryCXR/raw/MontgomerySet/
    CXR_png/MCUCXR_*.png                 — grayscale uint8, ~4000x4900, orientation varies
    ManualMask/leftMask/*.png            — boolean left-lung mask
    ManualMask/rightMask/*.png           — boolean right-lung mask

Montgomery ships no official split, so we make a deterministic one here
(SPLIT_SEED, 70/15/15).

Output: data/MontgomeryCXR/preprocessed_data/{train,val,test}/{idx:05d}/
  image.npy : float32 (256, 256, 1) HWC in [0, 1]   (raw /255)
  label.npy : uint8   (2, 256, 256) one-hot         (0 background, 1 lung = left | right)
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, '../../../data/MontgomeryCXR/raw/MontgomerySet')
TARGET_ROOT = os.path.join(HERE, '../../../data/MontgomeryCXR/preprocessed_data')

TARGET_SIZE = (256, 256)          # (W, H) for PIL.resize
SPLIT_SEED = 42
SPLIT_FRACS = (0.70, 0.15, 0.15)  # train / val / test
CLASS_NAMES = ['background', 'lung']


def list_ids():
    d = os.path.join(RAW, 'CXR_png')
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith('.png'))


def split_assignments(ids):
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(ids))
    n_tr = int(round(SPLIT_FRACS[0] * len(ids)))
    n_va = int(round(SPLIT_FRACS[1] * len(ids)))
    out = {}
    for rank, i in enumerate(order):
        out[ids[i]] = 'train' if rank < n_tr else ('val' if rank < n_tr + n_va else 'test')
    return out


def _resize_img(arr):
    im = Image.fromarray(arr).resize(TARGET_SIZE, Image.BILINEAR)
    return (np.asarray(im, dtype=np.float32) / 255.0)[..., None]        # (H, W, 1)


def _resize_mask(mask_bool):
    im = Image.fromarray(mask_bool.astype(np.uint8) * 255).resize(TARGET_SIZE, Image.NEAREST)
    return np.asarray(im) > 127


def process_one(sid, split):
    out_dir = os.path.join(TARGET_ROOT, split, f'{sid}')
    os.makedirs(out_dir, exist_ok=True)

    img = np.array(Image.open(os.path.join(RAW, 'CXR_png', f'{sid}.png')))
    if img.ndim == 3:
        img = img[..., 0]
    np.save(os.path.join(out_dir, 'image.npy'), _resize_img(img))

    left = np.array(Image.open(os.path.join(RAW, 'ManualMask', 'leftMask', f'{sid}.png'))) > 0
    right = np.array(Image.open(os.path.join(RAW, 'ManualMask', 'rightMask', f'{sid}.png'))) > 0
    lung = _resize_mask(left | right)
    label = np.stack([~lung, lung]).astype(np.uint8)                    # (2, H, W)
    np.save(os.path.join(out_dir, 'label.npy'), label)


def main():
    ids = list_ids()
    assign = split_assignments(ids)
    counts = {'train': 0, 'val': 0, 'test': 0}
    for k, sid in enumerate(ids):
        process_one(sid, assign[sid])
        counts[assign[sid]] += 1
        if (k + 1) % 25 == 0:
            print(f'{k + 1}/{len(ids)}  {counts}', flush=True)
    print(f'DONE  {counts}  -> {os.path.abspath(TARGET_ROOT)}', flush=True)


if __name__ == '__main__':
    main()
