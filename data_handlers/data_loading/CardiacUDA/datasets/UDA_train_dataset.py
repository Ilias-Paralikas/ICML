import torch
from torch.utils.data import Dataset
import os
import random

from .labels import remap_sample


class UDATrainDataset(Dataset):
    """UDA training dataset — every slice from the requested sites.

    Iterates over all slices in the sliced_data/train layout. A sample that has a
    saved label file is considered labeled and returns its real label; a sample
    without a label file returns a zero label.

    Labels are returned in the reduced ``train_channels`` space (see
    ``labels.remap_labels``): channel 0 is always the background, every other
    channel is one of the requested cardiac chambers, and all dropped channels
    are folded into the background.

    No labeled/unlabeled partition happens here — that is the job of
    SemiSupervisedDatasetWrapper.
    """

    def __init__(self,
                 root='../data_handlers/data/CardiacUDA/sliced_data/train',
                 train_augmentations=None,
                 num_classes=5,
                 train_channels=None,
                 sites=None):
        """
        num_classes     : raw one-hot channel count of the stored labels (default 5).
        train_channels  : raw channel indices to keep, e.g. [0, 3, 4]. If None,
                          all channels are kept.
        sites           : list of site names to include (e.g. ['Site_R_52', 'Site_G_100']).
                          If None, all sites are used.
        """
        self.root = root
        self.train_augmentations = train_augmentations
        self.num_classes = num_classes
        self.train_channels = train_channels if train_channels is not None else list(range(num_classes))

        self.data = []

        all_sites = sorted(os.listdir(root))
        if sites is not None:
            all_sites = [s for s in all_sites if s in sites]
        site_folders = [os.path.join(root, s) for s in all_sites]
        for site in site_folders:
            for patient in sorted(os.listdir(site)):
                patient_folder = os.path.join(site, patient)
                for slice_name in sorted(os.listdir(patient_folder)):
                    slice_folder = os.path.join(patient_folder, slice_name)
                    data_file = os.path.join(slice_folder, 'slice.pt')
                    label_file = os.path.join(slice_folder, 'label.pt')
                    has_label = os.path.exists(label_file)
                    self.data.append((data_file, label_file if has_label else None, has_label))

        self.labeled_indices   = [i for i, (_, _, flag) in enumerate(self.data) if flag]
        self.unlabeled_indices = [i for i, (_, _, flag) in enumerate(self.data) if not flag]
        print(f'UDATrainDataset: {len(self.data)} total samples '
              f'({len(self.labeled_indices)} labeled, {len(self.unlabeled_indices)} unlabeled) '
              f'| sites: {all_sites}')

    def __len__(self):
        return len(self.data)

    def _load_label(self, label_file):
        label = torch.load(label_file, weights_only=False).clone().float()
        if label.shape[0] < self.num_classes:
            pad = torch.zeros(self.num_classes - label.shape[0], *label.shape[1:])
            label = torch.cat([label, pad], dim=0)
        return remap_sample(label, self.train_channels)

    def __getitem__(self, idx):
        data_file, label_file, is_labeled = self.data[idx]
        image = torch.load(data_file, weights_only=False).clone()

        if self.train_augmentations is not None:
            image = self.train_augmentations(image)

        if is_labeled:
            label = self._load_label(label_file)
        else:
            h, w = image.shape[-2], image.shape[-1]
            label = torch.zeros(len(self.train_channels), h, w)

        return image, label, torch.tensor(is_labeled, dtype=torch.bool)

