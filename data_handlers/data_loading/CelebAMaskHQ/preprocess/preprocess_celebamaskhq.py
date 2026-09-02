"""Preprocess the official CelebAMask-HQ release into this project's standard
per-sample-folder layout, matching Camus's preprocessed_data/{train,val,test}/ pattern.

Source (see data_handlers/data_loading/CelebAMaskHQ/... README for provenance):
  original_data/archive (1)/CelebAMask-HQ/
    CelebA-HQ-img/{idx}.jpg                              — 1024x1024 RGB, idx 0-29999
    CelebAMask-HQ-mask-anno/{idx//2000}/{idx:05d}_{attr}.png  — 512x512 binary (0/255)
    CelebA-HQ-to-CelebA-mapping.txt                      — idx -> orig_idx (CelebA index)

LABEL SCHEMES
-------------
CelebAMask-HQ ships 18 raw, overlapping binary attribute masks. This script consolidates
them into a small one-hot label whose class set is chosen by a named *scheme* (``SCHEMES``
below). Every scheme is an explicit, intentional mapping: each output class lists exactly
which raw attributes are OR'd into it, and any raw attribute not named by any class falls
through to background (0). Merging (e.g. folding eyes / glasses into a single "face"
class) is therefore always deliberate and visible in one place — the script prints the
full resolved mapping (including what lands in background) before it runs.

Currently defined:
  full8 — the original 7 foreground parts + background, kept so runs on the old class set
          stay reproducible:
            1 skin      <- skin
            2 eyebrows  <- l_brow | r_brow
            3 eyes      <- l_eye | r_eye
            4 ears      <- l_ear | r_ear
            5 nose      <- nose
            6 mouth     <- mouth | u_lip | l_lip
            7 hair      <- hair
          (background: eye_g, ear_r, hat, neck, neck_l, cloth)
  fdm3  — the 3-class {background, face, hair} protocol used by Factorized Diffusion
          Models / DatasetDDPM, so our numbers are comparable to theirs:
            1 face  <- skin | nose | l_eye | r_eye | l_brow | r_brow | l_ear | r_ear
                       | mouth | u_lip | l_lip | eye_g   (the whole aligned face region,
                       glasses included)
            2 hair  <- hair
          (background: ear_r, hat, neck, neck_l, cloth)

To add a scheme: add an entry to ``SCHEMES`` and re-run. Output for scheme ``<s>`` goes to
its own folder ``data/CelebAMaskHQ/preprocessed_data_<s>/`` — existing preprocessed
folders (including the original ``preprocessed_data/``) are never overwritten.

Split: README says "we use the same train/val/test split as the CelebA dataset" —
applied via the mapping file's orig_idx against CelebA's well-known partition
boundaries (train/val/test = 162770/19867/19962 images, summing to CelebA's full
202599). This is the same split convention used elsewhere in the face-parsing
literature building on this release.

Output per sample: preprocessed_data_<scheme>/{train,val,test}/{idx:05d}/{image.npy,label.npy}
  image.npy : (256, 256, 3) float32 RGB in [0, 1] — scaled here (raw jpg /255) so the
              stored range matches CAMUS and CardiacUDA, whose preprocessed images are
              also float32 [0, 1] (verified: their loaders do no normalization, and the
              model's reconstruction head is sigmoid, so the target must be in [0, 1]).
              HWC layout — the dataset class permutes to (3, H, W) at load.
  label.npy : (NUM_CLASSES, 256, 256) uint8 one-hot — channel 0 = background, where
              NUM_CLASSES = 1 + (number of foreground classes in the chosen scheme).

Usage:
  python preprocess_celebamaskhq.py <scheme> [start] [end]
    <scheme>  required — one of SCHEMES (no default: the class set must be picked on
              purpose, since it decides which parts get merged).
    start/end optional — idx range to process, defaults 0..30000.
"""
import argparse
import os
import numpy as np
from PIL import Image

SOURCE_ROOT = '../../../data/CelebAMaskHQ/original_data/archive (1)/CelebAMask-HQ'
IMG_DIR = os.path.join(SOURCE_ROOT, 'CelebA-HQ-img')
MASK_DIR = os.path.join(SOURCE_ROOT, 'CelebAMask-HQ-mask-anno')
MAPPING_PATH = os.path.join(SOURCE_ROOT, 'CelebA-HQ-to-CelebA-mapping.txt')
# scheme <s> -> data/CelebAMaskHQ/preprocessed_data_<s>/  (the original, schemeless
# preprocessed_data/ folder is left alone).
TARGET_ROOT_TMPL = '../../../data/CelebAMaskHQ/preprocessed_data_{scheme}'

TARGET_SIZE = (256, 256)

# Every raw attribute mask CelebAMask-HQ ships (the {attr} in the mask filenames).
# Used to validate scheme definitions and to report what falls through to background.
KNOWN_ATTRS = {
    'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
    'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat',
}

# Named label schemes. Each is:
#   'classes'  : {class_idx (1..N): (human_name, [raw attrs OR'd into this class])}
#   'priority' : class indices in the order build_label() paints them — the raw masks
#                overlap at boundaries (hair/ear, skin/eye, ...), so a pixel claimed by
#                several classes goes to whichever appears LATER here. Broad/loosely
#                bounded regions first, precise features last so they win their edges.
# Any raw attr in KNOWN_ATTRS not listed by any class -> background (0).
SCHEMES = {
    'full8': {
        'classes': {
            1: ('skin',     ['skin']),
            2: ('eyebrows', ['l_brow', 'r_brow']),
            3: ('eyes',     ['l_eye', 'r_eye']),
            4: ('ears',     ['l_ear', 'r_ear']),
            5: ('nose',     ['nose']),
            6: ('mouth',    ['mouth', 'u_lip', 'l_lip']),
            7: ('hair',     ['hair']),
        },
        'priority': [1, 7, 4, 2, 5, 6, 3],
    },
    'fdm3': {
        'classes': {
            1: ('face', ['skin', 'nose', 'l_eye', 'r_eye', 'l_brow', 'r_brow',
                         'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'eye_g']),
            2: ('hair', ['hair']),
        },
        'priority': [1, 2],
    },
}

# CelebA's official partition boundaries (0-based orig_idx), from list_eval_partition.txt
TRAIN_END = 162770                  # [0, 162770)      -> train
VAL_END = TRAIN_END + 19867         # [162770, 182637) -> val
                                     # [182637, 202599) -> test


def validate_scheme(scheme):
    """Fail loudly on a malformed scheme rather than silently mislabelling 30k images."""
    classes, priority = scheme['classes'], scheme['priority']
    idxs = sorted(classes)
    assert idxs == list(range(1, len(idxs) + 1)), \
        f'class indices must be 1..N with no gaps, got {idxs}'
    assert sorted(priority) == idxs, \
        f'priority {priority} must be a permutation of class indices {idxs}'
    seen = {}
    for idx, (name, attrs) in classes.items():
        for attr in attrs:
            assert attr in KNOWN_ATTRS, f'class {idx} ({name}): unknown raw attr {attr!r}'
            assert attr not in seen, \
                f'raw attr {attr!r} assigned to both {seen[attr]} and {name}'
            seen[attr] = name


def describe_scheme(name, scheme):
    """Print the fully resolved mapping — including what goes to background — so any
    merge is visible before a run starts."""
    classes = scheme['classes']
    used = set()
    lines = [f'label scheme {name!r}  ({1 + len(classes)} classes, channel 0 = background)']
    for idx in sorted(classes):
        cname, attrs = classes[idx]
        used.update(attrs)
        lines.append(f'  {idx} {cname:<9} <- {" | ".join(attrs)}')
    dropped = sorted(KNOWN_ATTRS - used)
    lines.append(f'  0 background <- {" | ".join(dropped) if dropped else "(nothing else)"}')
    lines.append(f'  overlap priority (later wins): {scheme["priority"]}')
    print('\n'.join(lines), flush=True)


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


def build_label(idx, scheme):
    """(NUM_CLASSES, 512, 512) one-hot at native mask resolution, then resized.

    The 18 raw attribute masks are NOT mutually exclusive at their boundaries (e.g.
    hair/ear or skin/eye edges genuinely overlap by a pixel or two in the original
    hand annotations), so a plain per-class OR would double-count boundary pixels and
    break the one-hot convention the rest of the codebase relies on (remap_labels,
    argmax eval, ...). Resolved by painting classes in ``scheme['priority']`` order
    into a single label map; a pixel claimed by several classes keeps whichever was
    painted last.
    """
    classes = scheme['classes']
    num_classes = 1 + len(classes)

    class_map = np.zeros((512, 512), dtype=np.uint8)  # 0 = background by default
    for class_idx in scheme['priority']:
        _name, attrs = classes[class_idx]
        mask = np.zeros((512, 512), dtype=bool)
        for attr in attrs:
            mask |= load_attr_mask(idx, attr)
        class_map[mask] = class_idx  # later classes in the priority order win ties

    label = np.zeros((num_classes, 512, 512), dtype=bool)
    for c in range(num_classes):
        label[c] = class_map == c

    resized = np.zeros((num_classes, *TARGET_SIZE), dtype=np.uint8)
    for c in range(num_classes):
        # nearest-neighbor (NEAREST) keeps mask edges crisp — same reasoning as
        # preprocess_camus.ipynb's save_preproced_label.
        img = Image.fromarray(label[c].astype(np.uint8) * 255)
        img = img.resize(TARGET_SIZE, Image.NEAREST)
        resized[c] = (np.array(img) > 127).astype(np.uint8)
    return resized


def process_one(idx, split, scheme, target_root):
    out_dir = os.path.join(target_root, split, f'{idx:05d}')
    os.makedirs(out_dir, exist_ok=True)

    image = Image.open(os.path.join(IMG_DIR, f'{idx}.jpg')).convert('RGB')
    image = image.resize(TARGET_SIZE, Image.BILINEAR)
    # float32 in [0, 1] to match CAMUS / CardiacUDA's stored range (see module docstring).
    image = np.asarray(image, dtype=np.float32) / 255.0
    np.save(os.path.join(out_dir, 'image.npy'), image)

    label = build_label(idx, scheme)
    np.save(os.path.join(out_dir, 'label.npy'), label)


def main(scheme_name, start=0, end=30000, log_every=500):
    scheme = SCHEMES[scheme_name]
    validate_scheme(scheme)
    describe_scheme(scheme_name, scheme)

    target_root = TARGET_ROOT_TMPL.format(scheme=scheme_name)
    print(f'writing to {os.path.abspath(target_root)}', flush=True)

    mapping = load_mapping()
    counts = {'train': 0, 'val': 0, 'test': 0}
    for idx in range(start, end):
        split = split_for(mapping[idx])
        process_one(idx, split, scheme, target_root)
        counts[split] += 1
        if (idx + 1) % log_every == 0:
            print(f'{idx + 1}/{end} done | {counts}', flush=True)
    print(f'DONE {start}-{end} | {counts}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('scheme', choices=sorted(SCHEMES),
                        help='label scheme (which raw attrs merge into which class)')
    parser.add_argument('start', nargs='?', type=int, default=0)
    parser.add_argument('end', nargs='?', type=int, default=30000)
    args = parser.parse_args()
    main(args.scheme, args.start, args.end)
