import os

import numpy as np
import torch
from torch.utils.data import Dataset

from ...label_utils import remap_sample


class CelebAMaskHQDataset(Dataset):
    """CelebAMask-HQ face-parsing dataset (preprocessed).

    Reads the per-sample folders written by
    ``preprocess/preprocess_celebamaskhq.py``:
        preprocessed_data_<scheme>/{train,val,test}/{idx:05d}/{image.npy,label.npy}
      image.npy : (256, 256, 3) float32 RGB in [0, 1]   (HWC on disk)
      label.npy : (C, 256, 256) uint8 one-hot           (channel 0 = background)
    where C and the meaning of each channel depend on the preprocess label scheme
    (``full8`` -> 8 channels: skin/eyebrows/eyes/ears/nose/mouth/hair; ``fdm3`` -> 3:
    face/hair). ``root`` in dataset_config.json points at the folder for the scheme
    in use, so the channel count is read from the data, not assumed here.

    Structurally this is the "no anomaly" dataset, exactly like CamusDataset:
    every sample has ground truth, so ``is_labeled`` is always True and the
    semi-supervised split is left entirely to ``SemiSupervisedDatasetWrapper``
    (applied in build_dataset.py). ``return_is_labeled`` just selects the tuple
    shape for the two dataset "roles" used across this codebase:
      - train role (return_is_labeled=True):  (image, label, is_labeled)
      - eval  role (return_is_labeled=False): (image, label)

    Unlike the ultrasound datasets this one is RGB — ``image`` comes back as
    (3, H, W), so ``build_dataset`` reports ``in_channels=3`` for it (see there).
    """

    def __init__(self,
                 root='../data_handlers/data/CelebAMaskHQ/preprocessed_data',
                 mode='train',
                 train_augmentations=None,
                 train_channels=None,
                 return_is_labeled=True):
        """
        train_channels : raw channel indices to keep, e.g. [0, 1, 6, 7]. Every dropped
                         channel is folded back into the recomputed background
                         (channel 0) by label_utils.remap_sample — identical semantics
                         to CardiacUDA and Camus. None keeps every stored channel (the
                         count depends on the preprocess label scheme; see class doc).
        """
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

        # Stored channel count varies with the preprocess label scheme (full8 -> 8,
        # fdm3 -> 3, ...), so read it off the data rather than hardcoding it.
        first_label = np.load(os.path.join(self.sample_dirs[0], 'label.npy'), mmap_mode='r')
        self.raw_num_classes = first_label.shape[0]
        self.train_channels = (train_channels if train_channels is not None
                               else list(range(self.raw_num_classes)))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        # (H, W, 3) float32 [0, 1] on disk -> (3, H, W); already in [0, 1] to match
        # CAMUS / CardiacUDA (their loaders do no normalization), so NO /255 here.
        image = np.load(os.path.join(sample_dir, 'image.npy'))
        image = torch.from_numpy(image).float().permute(2, 0, 1).contiguous()

        # (C, H, W) uint8 one-hot -> reduced to train_channels, dropped channels
        # folded into a recomputed background 0 (same as the other datasets).
        label = torch.from_numpy(np.load(os.path.join(sample_dir, 'label.npy'))).float()
        label = remap_sample(label, self.train_channels)

        if self.train_augmentations is not None:
            image = self.train_augmentations(image)

        if self.return_is_labeled:
            is_labeled = torch.tensor(True, dtype=torch.bool)
            return image, label, is_labeled
        return image, label
