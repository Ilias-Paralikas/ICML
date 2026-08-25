
import torch
from torch.utils.data import Dataset
import random


class SemiSupervisedDatasetWrapper(Dataset):
    """Partition a dataset into a fixed labeled / unlabeled split.

    Takes any dataset and, seeded once, selects a fixed subset of its labeled
    samples to keep their labels. Every other sample is treated as unlabeled:
    its label is zeroed in __getitem__.

    The sampling pool is ``dataset.labeled_indices`` when the dataset exposes it.
    Otherwise it falls back to "all samples are labeled" (every index in the
    dataset), so plain supervised datasets can be converted to semi-supervised.

    Exposes labeled_indices / unlabeled_indices so loaders can be built with
    ``Subset(dataset, dataset.labeled_indices)``, etc.
    """

    def __init__(self, dataset, labeled_fraction=0.1, seed=42):
        self.dataset = dataset
        self.labeled_fraction = labeled_fraction

        pool = getattr(dataset, 'labeled_indices', None)
        if pool is None:
            pool = list(range(len(dataset)))   # fallback: every sample is labeled

        if len(pool) == 0:
            self._labeled_set = set()
        else:
            rng = random.Random(seed)
            k = max(1, int(len(pool) * labeled_fraction))
            k = min(k, len(pool))
            self._labeled_set = set(rng.sample(pool, k))

        self.labeled_indices   = [i for i in range(len(dataset)) if i in self._labeled_set]
        self.unlabeled_indices = [i for i in range(len(dataset)) if i not in self._labeled_set]
        print(f'SemiSupervisedDatasetWrapper: {len(dataset)} total samples '
              f'({len(self.labeled_indices)} labeled, {len(self.unlabeled_indices)} unlabeled)')
 
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label, is_labeled = self.dataset[idx]
        if idx not in self._labeled_set:
            label = torch.zeros_like(label)
            is_labeled = torch.tensor(False, dtype=torch.bool)
        return image, label, is_labeled

