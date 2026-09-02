"""Preprocess CHAOS **CT** (liver segmentation) into this repo's standard
per-sample-folder layout. One output folder per axial slice.

Raw (see ../chaos_explore.ipynb):
  data/CHAOS/raw/Train_Sets/CT/<pid>/
    DICOM_anon/i####,####b.dcm      — 512x512 uint16, HU = px*RescaleSlope + RescaleIntercept
    Ground/liver_GT_###.png        — boolean liver mask
  DICOM sorted BY FILENAME pairs index-for-index with Ground sorted by filename
  (the documented CHAOS correspondence; SimpleITK's geometric order is reversed).

Only Train_Sets has public labels. Split is patient-level (SPLIT_SEED, 70/15/15)
so no slices from one patient leak across splits.

CT window: abdominal soft-tissue, HU clipped to [WMIN, WMAX] then scaled to [0, 1].

Output: data/CHAOS/preprocessed_data_ct/{train,val,test}/{pid}_{sidx:04d}/
  image.npy : float32 (256, 256, 1) HWC in [0, 1]
  label.npy : uint8   (2, 256, 256) one-hot   (0 background, 1 liver)
"""
import glob
import os

import numpy as np
import pydicom
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CT_ROOT = os.path.join(HERE, '../../../data/CHAOS/raw/Train_Sets/CT')
TARGET_ROOT = os.path.join(HERE, '../../../data/CHAOS/preprocessed_data_ct')

TARGET_SIZE = (256, 256)
WMIN, WMAX = -160.0, 240.0        # abdominal window (level 40, width 400)
SPLIT_SEED = 42
SPLIT_FRACS = (0.70, 0.15, 0.15)
CLASS_NAMES = ['background', 'liver']


def _patients():
    return sorted((d for d in os.listdir(CT_ROOT) if os.path.isdir(os.path.join(CT_ROOT, d))),
                  key=int)


def _patient_split():
    pids = _patients()
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(pids))
    n_tr = int(round(SPLIT_FRACS[0] * len(pids)))
    n_va = int(round(SPLIT_FRACS[1] * len(pids)))
    out = {}
    for rank, i in enumerate(order):
        out[pids[i]] = 'train' if rank < n_tr else ('val' if rank < n_tr + n_va else 'test')
    return out


def _window(hu):
    x = np.clip(hu, WMIN, WMAX)
    return ((x - WMIN) / (WMAX - WMIN)).astype(np.float32)


def _read_hu(path):
    ds = pydicom.dcmread(path)
    a = ds.pixel_array.astype(np.float32)
    return a * float(getattr(ds, 'RescaleSlope', 1)) + float(getattr(ds, 'RescaleIntercept', 0))


def _resize_img(a):
    im = Image.fromarray((a * 255).astype(np.uint8)).resize(TARGET_SIZE, Image.BILINEAR)
    return (np.asarray(im, dtype=np.float32) / 255.0)[..., None]


def _resize_mask(m):
    im = Image.fromarray(m.astype(np.uint8) * 255).resize(TARGET_SIZE, Image.NEAREST)
    return np.asarray(im) > 127


def main():
    split = _patient_split()
    counts = {'train': 0, 'val': 0, 'test': 0}
    for pid in _patients():
        dcm = sorted(glob.glob(os.path.join(CT_ROOT, pid, 'DICOM_anon', '*.dcm')))
        gt = sorted(glob.glob(os.path.join(CT_ROOT, pid, 'Ground', '*.png')))
        n = min(len(dcm), len(gt))
        sp = split[pid]
        for i in range(n):
            img = _resize_img(_window(_read_hu(dcm[i])))
            m = np.array(Image.open(gt[i]))
            if m.ndim == 3:
                m = m[..., 0]
            liver = _resize_mask(m > 0)
            label = np.stack([~liver, liver]).astype(np.uint8)
            out_dir = os.path.join(TARGET_ROOT, sp, f'{pid}_{i:04d}')
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, 'image.npy'), img)
            np.save(os.path.join(out_dir, 'label.npy'), label)
            counts[sp] += 1
        print(f'patient {pid} [{sp}] {n} slices  running={counts}', flush=True)
    print(f'DONE  {counts}  -> {os.path.abspath(TARGET_ROOT)}', flush=True)


if __name__ == '__main__':
    main()
