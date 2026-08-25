
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from ...label_utils import remap_sample


class CamusDataset(Dataset):
    """CAMUS echocardiography dataset.

    Every sample has ground truth (no unlabeled slices the way CardiacUDA
    has), so `is_labeled` is always True. `return_is_labeled` only controls
    whether it's included in the returned tuple, matching the two dataset
    "roles" used elsewhere in this codebase:
      - train role (return_is_labeled=True):  (image, label, is_labeled) —
        the 3-tuple `SemiSupervisedDatasetWrapper` / `UDATrainDataset` use.
      - test role  (return_is_labeled=False): (image, label) — matching
        `UDATestDataset`.
    """

    def __init__(self,
                 root='../data_handlers/data/Camus/preprocessed_data',
                 mode='train',
                 train_augmentations=None,
                 frames=['ED', 'ES', 'half'],
                 views=['4CH'],
                 num_classes=4,
                 train_channels=None,
                 return_is_labeled=True):
        """
        num_classes     : raw one-hot channel count of the stored labels (always 4 for
                          CAMUS: background, LV endocardium, myocardium, left atrium).
        train_channels  : raw channel indices to keep, e.g. [0, 1, 3] to drop channel 2
                          (myocardium) — it gets folded into the background, same as
                          CardiacUDA's train_channels (see label_utils.remap_labels). If
                          None, all channels are kept.
        """
        self.root = root
        self.mode = mode
        self.train_augmentations = train_augmentations
        self.views = views
        self.frames = frames
        self.num_classes = num_classes
        self.train_channels = train_channels if train_channels is not None else list(range(num_classes))
        self.return_is_labeled = return_is_labeled

        self.mode_folder = os.path.join(root, mode)

        self.patient_dict = {}
        self.patient_list = []
        for patient in sorted(os.listdir(self.mode_folder)):
            self.patient_dict[patient] = {}
            patient_path = os.path.join(self.mode_folder, patient)
            for view in self.views:
                self.patient_dict[patient][view] = {}
                view_path = os.path.join(patient_path, view)
                for frame in self.frames:
                    frame_path = os.path.join(view_path, frame)
                    self.patient_dict[patient][view][frame] = []
                    slice_folders = sorted(os.listdir(frame_path))
                    for s in slice_folders:
                        slice_path = os.path.join(frame_path, s)
                        contents = os.listdir(slice_path)
                        for c in contents:
                            if '_gt' in c:
                                annotation_path = os.path.join(slice_path, c)
                            else:
                                image_path = os.path.join(slice_path, c)
                        self.patient_dict[patient][view][frame].append((image_path, annotation_path))
                        self.patient_list.append((image_path, annotation_path))
        self.len = len(self.patient_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image_path, annotation_path = self.patient_list[idx]

        # (H, W) -> (1, H, W): add the channel dim; the DataLoader adds the batch dim.
        image = torch.from_numpy(np.load(image_path)).float().unsqueeze(0)
        # already one-hot (C, H, W): channel 0 = background, 1-3 = LV endocardium /
        # myocardium / left atrium (see preprocess_camus.ipynb). Reduced to
        # self.train_channels, with every dropped channel folded back into background.
        label = torch.from_numpy(np.load(annotation_path)).float()
        label = remap_sample(label, self.train_channels)

        if self.train_augmentations is not None:
            image = self.train_augmentations(image)

        if self.return_is_labeled:
            is_labeled = torch.tensor(True, dtype=torch.bool)
            return image, label, is_labeled
        return image, label
