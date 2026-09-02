"""Preprocess the raw Drishti-GS1 release into this repo's standard
per-sample-folder layout (same as Camus / CelebAMask-HQ).

Raw (see ../drishtigs_explore.ipynb):
  data/DrishtiGS/raw/Drishti-GS1_files/Drishti-GS1_files/
    Training/Images/drishtiGS_*.png          + Training/GT/<id>/SoftMap/{*_ODsegSoftmap.png, *_cupsegSoftmap.png}
    Test/Images/...                           + Test/Test_GT/<id>/SoftMap/...
    .../GT/<id>/AvgBoundary/<id>_diskCenter.txt   — optic-disc centre, "row col"
  RGB fundus photos, uint8, ~1750x2050; softmaps are 0-255 (4-expert average).

DISC-CENTRED CROP (important): the raw fundus photo is a full posterior-pole shot
in which the optic disc is only ~3% of the frame and the cup ~1.5% — far too
small for a fixed-spatial-template method to latch onto (both foreground classes
end up near-zero area after the 256x256 resize). So, like every OD/OC segmentation
paper, we first crop a square patch around the disc: side = CROP_SCALE * (disc
bounding box, longer side), centred on `diskCenter.txt`, clamped to the image,
then resized to 256x256. At CROP_SCALE = 1.6 the disc is ~30% of the crop and the
cup ~15%. Set CROP_SCALE = None to disable the crop (old behaviour, full frame).

Split: Drishti's own 51-image Test set -> test; its 50-image Training set is
split here (SPLIT_SEED, 40/10) into train / val.

Labels are nested (cup subset of disc): class map 0 background, 1 disc rim,
2 cup (cup painted over disc), then one-hot.

Output: data/DrishtiGS/preprocessed_data/{train,val,test}/{idx:05d}/
  image.npy : float32 (256, 256, 3) HWC RGB in [0, 1]   (raw /255)
  label.npy : uint8   (3, 256, 256) one-hot             (0 background, 1 optic disc rim, 2 optic cup)
"""
import glob
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '../../../data/DrishtiGS/raw/Drishti-GS1_files/Drishti-GS1_files')
TARGET_ROOT = os.path.join(HERE, '../../../data/DrishtiGS/preprocessed_data')

TARGET_SIZE = (256, 256)
CROP_SCALE = 1.6                 # crop side = CROP_SCALE * disc-bbox longer side; None = no crop
SPLIT_SEED = 42
N_VAL = 10                       # of the 50 official-Training images, held out for val
CLASS_NAMES = ['background', 'optic disc', 'optic cup']


def _ids(img_dir):
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(img_dir, '*.png')))


def _plan():
    """(sid, img_dir, gt_dir, split) for every sample."""
    tr_img = os.path.join(BASE, 'Training', 'Images')
    tr_gt = os.path.join(BASE, 'Training', 'GT')
    te_img = os.path.join(BASE, 'Test', 'Images')
    te_gt = os.path.join(BASE, 'Test', 'Test_GT')

    tr_ids = _ids(tr_img)
    rng = np.random.default_rng(SPLIT_SEED)
    val_ids = set(rng.choice(tr_ids, size=N_VAL, replace=False).tolist())

    plan = []
    for sid in tr_ids:
        plan.append((sid, tr_img, tr_gt, 'val' if sid in val_ids else 'train'))
    for sid in _ids(te_img):
        plan.append((sid, te_img, te_gt, 'test'))
    return plan


def _softmap(gt_dir, sid, kind):
    p = os.path.join(gt_dir, sid, 'SoftMap', f'{sid}_{kind}.png')
    a = np.array(Image.open(p))
    return (a[..., 0] if a.ndim == 3 else a) > 128


def _disc_center(gt_dir, sid, disc):
    """(row, col) disc centre — from diskCenter.txt if present, else the disc-mask centroid."""
    cf = os.path.join(gt_dir, sid, 'AvgBoundary', f'{sid}_diskCenter.txt')
    if os.path.exists(cf):
        r, c = open(cf).read().split()[:2]
        return float(r), float(c)
    ys, xs = np.where(disc)
    return float(ys.mean()), float(xs.mean())


def _crop_box(disc, center, H, W):
    """Square crop (r0, r1, c0, c1) around `center`, side = CROP_SCALE * disc bbox,
    clamped to the image. Falls back to the full frame if the disc mask is empty."""
    ys, xs = np.where(disc)
    if len(xs) == 0:
        return 0, H, 0, W
    side = CROP_SCALE * max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    side = int(min(side, H, W))
    cy, cx = center
    r0 = int(np.clip(round(cy - side / 2), 0, H - side))
    c0 = int(np.clip(round(cx - side / 2), 0, W - side))
    return r0, r0 + side, c0, c0 + side


def process_one(sid, img_dir, gt_dir, split, k):
    out_dir = os.path.join(TARGET_ROOT, split, f'{k:05d}')
    os.makedirs(out_dir, exist_ok=True)

    img = np.array(Image.open(os.path.join(img_dir, f'{sid}.png')).convert('RGB'))
    disc = _softmap(gt_dir, sid, 'ODsegSoftmap')
    cup = _softmap(gt_dir, sid, 'cupsegSoftmap')
    H, W = disc.shape

    if CROP_SCALE:
        r0, r1, c0, c1 = _crop_box(disc, _disc_center(gt_dir, sid, disc), H, W)
        img = img[r0:r1, c0:c1]
        disc = disc[r0:r1, c0:c1]
        cup = cup[r0:r1, c0:c1]

    img = Image.fromarray(img).resize(TARGET_SIZE, Image.BILINEAR)
    np.save(os.path.join(out_dir, 'image.npy'), np.asarray(img, dtype=np.float32) / 255.0)

    cmap = np.where(disc, 1, 0)
    cmap = np.where(cup, 2, cmap).astype(np.uint8)                      # nested
    cmap = np.array(Image.fromarray(cmap).resize(TARGET_SIZE, Image.NEAREST))
    label = np.stack([(cmap == c) for c in range(len(CLASS_NAMES))]).astype(np.uint8)
    np.save(os.path.join(out_dir, 'label.npy'), label)


def main():
    plan = _plan()
    counts = {'train': 0, 'val': 0, 'test': 0}
    per_split_k = {'train': 0, 'val': 0, 'test': 0}
    for sid, img_dir, gt_dir, split in plan:
        process_one(sid, img_dir, gt_dir, split, per_split_k[split])
        per_split_k[split] += 1
        counts[split] += 1
    print(f'DONE  {counts}  crop_scale={CROP_SCALE}  -> {os.path.abspath(TARGET_ROOT)}', flush=True)


if __name__ == '__main__':
    main()
