"""Builds all of CardiacUDA's DataLoaders from this dataset's own JSON config
(dataset_config.json, next to this file).

This is the only CardiacUDA-specific glue code a caller needs to know about —
everything else (paths, channels, sites) lives in the JSON file, not in code. See
dataset_registry.py for why this function owns the full loader-building pipeline
(labeled_fraction/seed included) instead of just handing back raw datasets.
"""
import json
import random
from pathlib import Path

from torch.utils.data import DataLoader, Dataset, Subset

from .datasets import UDATrainDataset, UDATestDataset

CONFIG_PATH = Path(__file__).parent / 'dataset_config.json'


def _resolve(path_str):
    """Paths in dataset_config.json are relative to this file's own directory,
    not to whatever notebook/script happens to call build_dataset() — this
    keeps the config caller-cwd-independent."""
    return str((CONFIG_PATH.parent / path_str).resolve())


class _EvalPairs(Dataset):
    """Adapts a (image, label, is_labeled)-yielding dataset to the (image, label)
    2-tuple contract used for val_loader/test_loader elsewhere in this codebase."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label, _ = self.dataset[idx]
        return image, label


def build_dataset(batch_size=32, labeled_fraction=0.1, seed=42, eval_batch_size=16):
    """Build all of CardiacUDA's DataLoaders.

    ANOMALY, READ BEFORE TOUCHING: CardiacUDA does not use
    SemiSupervisedDatasetWrapper at all, even though it takes the same
    `labeled_fraction`/`seed` every other dataset's build_dataset() takes and gives
    them the same meaning: `labeled_fraction` of the labeled pool
    (`UDATrainDataset.labeled_indices`, ~735 slices) is kept for training (seeded via
    `seed`, same `max(1, int(pool * labeled_fraction))` floor SemiSupervisedDatasetWrapper
    itself uses) — everything else in that pool is NOT used for training at all.

    That "everything else" is where CardiacUDA diverges from Camus: Camus treats its
    unchosen labeled samples as unlabeled reconstruction data (via the wrapper);
    CardiacUDA instead holds them out COMPLETELY — real labels, as a genuine
    validation set, never touched by training in any form (not even unlabeled
    reconstruction). That's a deliberate choice, not a wrapper limitation — see the
    "no reconstruction leakage" requirement this was built for — and it's why this
    can't just be SemiSupervisedDatasetWrapper(train_dataset, labeled_fraction, seed):
    the wrapper's "everything not labeled" bucket IS its unlabeled_indices, which
    would put these samples back into training as reconstruction input. So instead
    this builds three disjoint index sets directly:
        - labeled   : labeled_fraction of the labeled pool — used as real-labeled
                      training data, no further split.
        - val       : the REST of the labeled pool — held out entirely (real labels,
                      never seen by training in any form).
        - unlabeled : dataset.unlabeled_indices — UDATrainDataset already returns a
                      zeroed label for these natively, so no wrapper/masking needed.
    Camus (see Camus/build_dataset.py) has no such requirement and uses
    SemiSupervisedDatasetWrapper normally.

    Returns:
        labeled_loader, unlabeled_loader, val_loader, test_loader, num_classes
        (val_loader is None if labeled_fraction leaves nothing for val, e.g. 1.0)
    """
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)

    full_train_dataset = UDATrainDataset(
        root=_resolve(cfg['train_root']),
        train_channels=cfg['train_channels'],
        sites=cfg['sites'],
    )

    labeled_pool = list(full_train_dataset.labeled_indices)
    rng = random.Random(seed)
    # Same floor as SemiSupervisedDatasetWrapper: at least 1 real-labeled training
    # sample as long as the pool isn't empty, regardless of how small labeled_fraction is.
    k = max(1, int(len(labeled_pool) * labeled_fraction)) if labeled_pool else 0
    k = min(k, len(labeled_pool))
    train_labeled_indices = rng.sample(labeled_pool, k)
    train_labeled_set = set(train_labeled_indices)
    val_indices = [i for i in labeled_pool if i not in train_labeled_set]

    print(f'CardiacUDA build_dataset: {len(labeled_pool)} labeled slices, '
          f'labeled_fraction={labeled_fraction} -> {len(train_labeled_indices)} kept for '
          f'training, {len(val_indices)} held out for val entirely (no unlabeled/'
          f'reconstruction use either); {len(full_train_dataset.unlabeled_indices)} '
          f'unlabeled slices unaffected by any of this')

    labeled_loader = DataLoader(
        Subset(full_train_dataset, train_labeled_indices),
        batch_size=batch_size, shuffle=True, num_workers=0)
    unlabeled_loader = DataLoader(
        Subset(full_train_dataset, full_train_dataset.unlabeled_indices),
        batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = None
    if val_indices:
        val_loader = DataLoader(
            _EvalPairs(Subset(full_train_dataset, sorted(val_indices))),
            batch_size=eval_batch_size, shuffle=True, num_workers=0)

    test_dataset = UDATestDataset(
        root=_resolve(cfg['test_root']),
        train_channels=cfg['train_channels'],
    )
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=True, num_workers=0)

    num_classes = len(cfg['train_channels'])   # one class per kept channel

    return labeled_loader, unlabeled_loader, val_loader, test_loader, num_classes
