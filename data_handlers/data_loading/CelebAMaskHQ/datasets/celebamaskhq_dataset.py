import os

import numpy as np
import torch
from torch.utils.data import Dataset

from ...label_utils import remap_sample


class CelebAMaskHQDataset(Dataset):
    """CelebAMask-HQ face-parsing dataset (preprocessed).

    Reads the per-sample folders written by
    ``preprocess/preprocess_celebamaskhq.py``:
        preprocessed_data/{train,val,test}/{idx:05d}/{image.npy,label.npy}
      image.npy : (256, 256, 3) float32 RGB in [0, 1]   (HWC on disk)
      label.npy : (8, 256, 256) uint8 one-hot            (channel 0 = background)

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

    # Raw stored label is 8-channel one-hot; see preprocess script for the mapping.
    #   0 background  1 skin  2 eyebrows  3 eyes  4 ears  5 nose  6 mouth  7 hair
    RAW_NUM_CLASSES = 8

    def __init__(self,
                 root='../data_handlers/data/CelebAMaskHQ/preprocessed_data',
                 mode='train',
                 train_augmentations=None,
                 train_channels=None,
                 return_is_labeled=True):
        """
        train_channels : raw channel indices to keep, e.g. [0, 1, 6, 7] to train on
                         only skin/mouth/hair. Every dropped channel is folded back
                         into the recomputed background (channel 0) by
                         label_utils.remap_sample — identical semantics to CardiacUDA
                         and Camus. None keeps all 8 channels.
        """
        self.root = root
        self.mode = mode
        self.train_augmentations = train_augmentations
        self.train_channels = (train_channels if train_channels is not None
                               else list(range(self.RAW_NUM_CLASSES)))
        self.return_is_labeled = return_is_labeled

        self.mode_folder = os.path.join(root, mode)
        self.sample_dirs = [
            os.path.join(self.mode_folder, name)
            for name in sorted(os.listdir(self.mode_folder))
        ]
        self.len = len(self.sample_dirs)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        # (H, W, 3) float32 [0, 1] on disk -> (3, H, W); already in [0, 1] to match
        # CAMUS / CardiacUDA (their loaders do no normalization), so NO /255 here.
        image = np.load(os.path.join(sample_dir, 'image.npy'))
        image = torch.from_numpy(image).float().permute(2, 0, 1).contiguous()

        # (8, H, W) uint8 one-hot -> reduced to train_channels, dropped channels
        # folded into a recomputed background 0 (same as the other datasets).
        label = torch.from_numpy(np.load(os.path.join(sample_dir, 'label.npy'))).float()
        label = remap_sample(label, self.train_channels)

        if self.train_augmentations is not None:
            image = self.train_augmentations(image)

        if self.return_is_labeled:
            is_labeled = torch.tensor(True, dtype=torch.bool)
            return image, label, is_labeled
        return image, label
