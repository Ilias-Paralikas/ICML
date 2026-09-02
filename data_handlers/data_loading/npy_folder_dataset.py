"""Generic per-sample-folder dataset for the preprocessed `.npy` layout.

Several datasets in this repo store their preprocessed data the exact same way —
``preprocessed_data/{train,val,test}/{idx}/{image.npy,label.npy}`` with

* ``image.npy`` : ``float32`` ``(H, W, C)`` HWC in ``[0, 1]``  (C = 1 grayscale, 3 RGB)
* ``label.npy`` : ``uint8``   ``(K, H, W)`` one-hot, channel 0 = background

and are otherwise identical "no anomaly" datasets (every sample has ground truth,
on-disk train/val/test split, semi-supervised split left to
``SemiSupervisedDatasetWrapper``). ``NpyFolderDataset`` is that shared body — used
by the ISIC 2017 / Montgomery / CHAOS-CT / Drishti-GS loaders. It mirrors
``CelebAMaskHQDataset`` (which predates this and is left as-is) but reads the image
channel count and raw class count off the data instead of hardcoding them.
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .label_utils import remap_sample


class NpyFolderDataset(Dataset):
    """Read ``root/{mode}/{idx}/{image.npy,label.npy}``.

    return_is_labeled selects the tuple shape for the two dataset "roles" used
    across this codebase:
      - train role (True):  (image, label, is_labeled)   — is_labeled always True
      - eval  role (False): (image, label)

    train_channels : raw label channel indices to keep (e.g. [0, 1]); every
        dropped channel is folded into a recomputed background 0 by
        ``label_utils.remap_sample`` — same semantics as every other dataset here.
        None keeps all stored channels.

    Exposes ``in_channels`` and ``raw_num_classes``, read from the first sample.
    """

    def __init__(self,
                 root,
                 mode='train',
                 train_augmentations=None,
                 train_channels=None,
                 return_is_labeled=True):
        self.root = root
        self.mode = mode
        self.train_augmentations = train_augmentations
        self.return_is_labeled = return_is_labeled

        self.mode_folder = os.path.join(root, mode)
        self.sample_dirs = [
            os.path.join(self.mode_folder, name)
            for name in sorted(os.listdir(self.mode_folder))
        ]
        self.len = len(self.sample_dirs)
        if self.len == 0:
            raise RuntimeError(f'no samples under {self.mode_folder}')

        img0 = np.load(os.path.join(self.sample_dirs[0], 'image.npy'), mmap_mode='r')
        lab0 = np.load(os.path.join(self.sample_dirs[0], 'label.npy'), mmap_mode='r')
        self.in_channels = img0.shape[2] if img0.ndim == 3 else 1
        self.raw_num_classes = lab0.shape[0]
        self.train_channels = (train_channels if train_channels is not None
                               else list(range(self.raw_num_classes)))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        image = np.load(os.path.join(sample_dir, 'image.npy'))          # (H, W, C) or (H, W)
        if image.ndim == 2:
            image = image[..., None]
        image = torch.from_numpy(np.ascontiguousarray(image)).float().permute(2, 0, 1).contiguous()

        label = torch.from_numpy(np.load(os.path.join(sample_dir, 'label.npy'))).float()
        label = remap_sample(label, self.train_channels)

        if self.train_augmentations is not None:
            image = self.train_augmentations(image)

        if self.return_is_labeled:
            return image, label, torch.tensor(True, dtype=torch.bool)
        return image, label
