"""Preprocess the official CelebAMask-HQ release into this project's standard
per-sample-folder layout, matching Camus's preprocessed_data/{train,val,test}/ pattern.

Source (see data_handlers/data_loading/CelebAMaskHQ/... README for provenance):
  original_data/archive (1)/CelebAMask-HQ/
    CelebA-HQ-img/{idx}.jpg                              — 1024x1024 RGB, idx 0-29999
    CelebAMask-HQ-mask-anno/{idx//2000}/{idx:05d}_{attr}.png  — 512x512 binary (0/255)
    CelebA-HQ-to-CelebA-mapping.txt                      — idx -> orig_idx (CelebA index)

Consolidates the 18 raw attributes down to 7 foreground classes + background (agreed
scope: only "on the aligned face" parts — accessories and neck/cloth are dropped,
folded into background, since they're off the landmark-aligned region and would
undermine the consistent-positioning property this dataset is being used to test):
    1 skin      <- skin
    2 eyebrows  <- l_brow | r_brow
    3 eyes      <- l_eye | r_eye
    4 ears      <- l_ear | r_ear
    5 nose      <- nose
    6 mouth     <- mouth | u_lip | l_lip
    7 hair      <- hair
    (dropped, folded into background: eye_g, ear_r, hat, neck, neck_l, cloth)

Split: README says "we use the same train/val/test split as the CelebA dataset" —
applied via the mapping file's orig_idx against CelebA's well-known partition
boundaries (train/val/test = 162770/19867/19962 images, summing to CelebA's full
202599). This is the same split convention used elsewhere in the face-parsing
literature building on this release.

Output per sample: preprocessed_data/{train,val,test}/{idx:05d}/{image.npy,label.npy}
  image.npy : (256, 256, 3) float32 RGB in [0, 1] — scaled here (raw jpg /255) so the
              stored range matches CAMUS and CardiacUDA, whose preprocessed images are
              also float32 [0, 1] (verified: their loaders do no normalization, and the
              model's reconstruction head is sigmoid-activated, so the target must be
              in [0, 1]). HWC layout — the dataset class permutes to (3, H, W) at load.
  label.npy : (8, 256, 256) uint8 one-hot — channel 0 = background
"""
import os
import sys
import numpy as np
from PIL import Image

SOURCE_ROOT = '../../../data/CelebAMaskHQ/original_data/archive (1)/CelebAMask-HQ'
IMG_DIR = os.path.join(SOURCE_ROOT, 'CelebA-HQ-img')
MASK_DIR = os.path.join(SOURCE_ROOT, 'CelebAMask-HQ-mask-anno')
MAPPING_PATH = os.path.join(SOURCE_ROOT, 'CelebA-HQ-to-CelebA-mapping.txt')
TARGET_ROOT = '../../../data/CelebAMaskHQ/preprocessed_data'

TARGET_SIZE = (256, 256)

# class index (1-7) -> raw attribute names folded into it (OR'd together)
CLASS_ATTRS = {
    1: ['skin'],
    2: ['l_brow', 'r_brow'],
    3: ['l_eye', 'r_eye'],
    4: ['l_ear', 'r_ear'],
    5: ['nose'],
    6: ['mouth', 'u_lip', 'l_lip'],
    7: ['hair'],
}
NUM_CLASSES = 8  # background (0) + 7 above

# CelebA's official partition boundaries (0-based orig_idx), from list_eval_partition.txt
TRAIN_END = 162770                  # [0, 162770)      -> train
VAL_END = TRAIN_END + 19867         # [162770, 182637) -> val
                                     # [182637, 202599) -> test


def split_for(orig_idx):
    if orig_idx < TRAIN_END:
        return 'train'
    elif orig_idx < VAL_END:
        return 'val'
    else:
        return 'test'


def load_mapping():
    """idx -> orig_idx, from the mapping file's whitespace-separated columns."""
    mapping = {}
    with open(MAPPING_PATH) as f:
        next(f)  # header
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            idx, orig_idx = int(parts[0]), int(parts[1])
            mapping[idx] = orig_idx
    return mapping


def load_attr_mask(idx, attr):
    """Load one raw attribute mask if it exists, else an all-zero mask (attribute
    absent for this image is common — e.g. ears/accessories not visible)."""
    folder = str(idx // 2000)
    path = os.path.join(MASK_DIR, folder, f'{idx:05d}_{attr}.png')
    if not os.path.exists(path):
        return np.zeros((512, 512), dtype=bool)
    arr = np.array(Image.open(path).convert('L'))
    return arr > 127


# The 18 raw attribute masks are NOT mutually exclusive at their boundaries (e.g.
# hair/ear or skin/eye edges genuinely overlap by a pixel or two in the original
# hand annotations) — so a plain per-class OR would double-count boundary pixels and
# break the one-hot convention every other dataset in this codebase relies on
# (remap_labels, argmax-based eval, ...). Resolved here by assigning each pixel to
# exactly one class via a fixed priority order: broad/catch-all regions first, small
# precise features layered on top and winning any overlap — skin and hair are the
# least precisely-bounded masks, so they go first and get overwritten at their edges
# by whichever more specific feature actually claims that pixel.
CLASS_PRIORITY = [1, 7, 4, 2, 5, 6, 3]  # skin, hair, ears, eyebrows, nose, mouth, eyes
assert set(CLASS_PRIORITY) == set(CLASS_ATTRS)


def build_label(idx):
    """(NUM_CLASSES, 512, 512) one-hot at native mask resolution, then resized."""
    class_map = np.zeros((512, 512), dtype=np.uint8)  # 0 = background by default
    for class_idx in CLASS_PRIORITY:
        mask = np.zeros((512, 512), dtype=bool)
        for attr in CLASS_ATTRS[class_idx]:
            mask |= load_attr_mask(idx, attr)
        class_map[mask] = class_idx  # later classes in the priority order win ties

    label = np.zeros((NUM_CLASSES, 512, 512), dtype=bool)
    for c in range(NUM_CLASSES):
        label[c] = class_map == c

    resized = np.zeros((NUM_CLASSES, *TARGET_SIZE), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        # nearest-neighbor (NEAREST) keeps mask edges crisp — same reasoning as
        # preprocess_camus.ipynb's save_preproced_label.
        img = Image.fromarray(label[c].astype(np.uint8) * 255)
        img = img.resize(TARGET_SIZE, Image.NEAREST)
        resized[c] = (np.array(img) > 127).astype(np.uint8)
    return resized


def process_one(idx, split):
    out_dir = os.path.join(TARGET_ROOT, split, f'{idx:05d}')
    os.makedirs(out_dir, exist_ok=True)

    image = Image.open(os.path.join(IMG_DIR, f'{idx}.jpg')).convert('RGB')
    image = image.resize(TARGET_SIZE, Image.BILINEAR)
    # float32 in [0, 1] to match CAMUS / CardiacUDA's stored range (see module docstring).
    image = np.asarray(image, dtype=np.float32) / 255.0
    np.save(os.path.join(out_dir, 'image.npy'), image)

    label = build_label(idx)
    np.save(os.path.join(out_dir, 'label.npy'), label)


def main(start=0, end=30000, log_every=500):
    mapping = load_mapping()
    counts = {'train': 0, 'val': 0, 'test': 0}
    for idx in range(start, end):
        split = split_for(mapping[idx])
        process_one(idx, split)
        counts[split] += 1
        if (idx + 1) % log_every == 0:
            print(f'{idx + 1}/{end} done | {counts}', flush=True)
    print(f'DONE {start}-{end} | {counts}', flush=True)


if __name__ == '__main__':
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
    main(start, end)
