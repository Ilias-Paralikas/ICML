"""Preprocess the raw ISIC 2017 (Part 1, lesion segmentation) release into this
repo's standard per-sample-folder layout (same as Camus / CelebAMask-HQ).

Raw (see ../isic2017_explore.ipynb):
  data/ISIC2017/raw/
    ISIC-2017_Training_Data/ISIC_*.jpg          + ISIC-2017_Training_Part1_GroundTruth/*_segmentation.png
    ISIC-2017_Validation_Data/...               + ISIC-2017_Validation_Part1_GroundTruth/...
    ISIC-2017_Test_v2_Data/...                  + ISIC-2017_Test_v2_Part1_GroundTruth/...
  RGB dermoscopy photos, uint8, native resolution varies a lot; masks are 0/255 PNG.
  (`*_superpixels.png` files in the *_Data folders are ignored.)

Uses ISIC's own train / validation / test split.

Output: data/ISIC2017/preprocessed_data/{train,val,test}/{idx:05d}/
  image.npy : float32 (256, 256, 3) HWC RGB in [0, 1]   (raw /255)
  label.npy : uint8   (2, 256, 256) one-hot             (0 background, 1 lesion)
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, '../../../data/ISIC2017/raw')
TARGET_ROOT = os.path.join(HERE, '../../../data/ISIC2017/preprocessed_data')

TARGET_SIZE = (256, 256)
CLASS_NAMES = ['background', 'lesion']

# split name -> (image dir, ground-truth dir), all relative to RAW
SPLITS = {
    'train': ('ISIC-2017_Training_Data',   'ISIC-2017_Training_Part1_GroundTruth'),
    'val':   ('ISIC-2017_Validation_Data', 'ISIC-2017_Validation_Part1_GroundTruth'),
    'test':  ('ISIC-2017_Test_v2_Data',    'ISIC-2017_Test_v2_Part1_GroundTruth'),
}


def _ids(img_dir):
    return sorted(os.path.basename(p)[:-4]
                  for p in glob.glob(os.path.join(RAW, img_dir, 'ISIC_*.jpg')))


def _resize_img(arr):
    im = Image.fromarray(arr).convert('RGB').resize(TARGET_SIZE, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0                     # (H, W, 3)


def _resize_mask(arr):
    im = Image.fromarray(arr).resize(TARGET_SIZE, Image.NEAREST)
    return np.asarray(im) > 127


def process_one(sid, split, img_dir, gt_dir, k):
    out_dir = os.path.join(TARGET_ROOT, split, f'{k:05d}')
    os.makedirs(out_dir, exist_ok=True)

    img = np.array(Image.open(os.path.join(RAW, img_dir, f'{sid}.jpg')))
    np.save(os.path.join(out_dir, 'image.npy'), _resize_img(img))

    m = np.array(Image.open(os.path.join(RAW, gt_dir, f'{sid}_segmentation.png')))
    if m.ndim == 3:
        m = m[..., 0]
    lesion = _resize_mask(m)
    label = np.stack([~lesion, lesion]).astype(np.uint8)               # (2, H, W)
    np.save(os.path.join(out_dir, 'label.npy'), label)


def main(only=None):
    for split, (img_dir, gt_dir) in SPLITS.items():
        if only and split != only:
            continue
        ids = _ids(img_dir)
        if not ids:
            print(f'[{split}] no images under {img_dir} — skipped', flush=True)
            continue
        for k, sid in enumerate(ids):
            process_one(sid, split, img_dir, gt_dir, k)
            if (k + 1) % 200 == 0:
                print(f'[{split}] {k + 1}/{len(ids)}', flush=True)
        print(f'[{split}] DONE {len(ids)}', flush=True)
    print(f'-> {os.path.abspath(TARGET_ROOT)}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
