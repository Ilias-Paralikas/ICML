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


def build_dataset(batch_size=32, labeled_fraction=0.1, seed=42, eval_batch_size=16,
                   dataset_config_path=None):
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
    this builds three (or four, see `same_site_test` below) disjoint index sets
    directly:
        - labeled   : labeled_fraction of the labeled pool — used as real-labeled
                      training data, no further split.
        - val (+test) : the REST of the labeled pool — held out entirely (real
                      labels, never seen by training in any form).
        - unlabeled : dataset.unlabeled_indices — UDATrainDataset already returns a
                      zeroed label for these natively, so no wrapper/masking needed.
    Camus (see Camus/build_dataset.py) has no such requirement and uses
    SemiSupervisedDatasetWrapper normally.

    `same_site_test` (dataset_config.json): CardiacUDA's real `UDATestDataset` (built
    from `test_root`) is a genuine cross-site holdout — a different acquisition site
    than every training/val slice (see `data/CardiacUDA/original_data/{train,test}` —
    train draws from `Site_G_*`/`Site_R_*`, test is entirely a separate `Site_test`).
    That's a domain-shift evaluation, which is the wrong thing to measure for a method
    that isn't a domain-adaptation framework and isn't claiming cross-site robustness.
    When `same_site_test` is `true`, `test_root`/`UDATestDataset` are ignored entirely
    and instead the "REST of the labeled pool" above (what would otherwise be the
    whole val set) is itself split in half — first half stays val, second half becomes
    test — so train, val, and test are all drawn from the same (training) sites.
    Both halves are seeded off the same `rng` as the labeled/val split above, so the
    whole train/val/test partition is reproducible from `seed` alone.

    `max_unlabeled_samples` (dataset_config.json): `false` (default) keeps every
    natively-unlabeled slice (~16.5k) for reconstruction training, same as always. Set
    to an int to instead use a fixed, seeded random subset of at most that many —
    e.g. for a quick run, or to study how much unlabeled data the reconstruction path
    actually needs. Sampled off the same `rng` as everything else above, so it's part
    of the same reproducible-from-`seed` partition.

    dataset_config_path : read the dataset config from here instead of the live
                          CONFIG_PATH next to this file. Path resolution for entries
                          inside it (train_root, test_root) is still anchored on
                          CONFIG_PATH's own directory regardless — only which file's
                          *content* gets read changes. Used to reproduce an old run
                          exactly from a saved snapshot; see
                          dataset_registry.get_dataset_config_path and the training
                          notebook's LOAD_RUN handling.

    Returns:
        labeled_loader, unlabeled_loader, val_loader, test_loader, num_classes, in_channels
        (val_loader/test_loader are None if there's nothing left to put in them —
        e.g. labeled_fraction=1.0 leaves no val, or same_site_test halves a val pool
        of size 0 or 1)
    """
    with open(dataset_config_path or CONFIG_PATH) as f:
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
    held_out_indices = [i for i in labeled_pool if i not in train_labeled_set]

    same_site_test = cfg.get('same_site_test', False)
    if same_site_test:
        # Continue the same seeded rng stream — the whole labeled/val/test partition
        # is reproducible from `seed` alone, no separate seed needed for this split.
        held_out_indices = held_out_indices.copy()
        rng.shuffle(held_out_indices)
        half = len(held_out_indices) // 2
        val_indices, test_indices = held_out_indices[:half], held_out_indices[half:]
    else:
        val_indices, test_indices = held_out_indices, None

    unlabeled_indices = full_train_dataset.unlabeled_indices
    max_unlabeled_samples = cfg.get('max_unlabeled_samples', False)
    if max_unlabeled_samples is not False:
        n = max(0, min(int(max_unlabeled_samples), len(unlabeled_indices)))
        unlabeled_indices = rng.sample(unlabeled_indices, n)

    print(f'CardiacUDA build_dataset: {len(labeled_pool)} labeled slices, '
          f'labeled_fraction={labeled_fraction} -> {len(train_labeled_indices)} kept for '
          f'training, {len(held_out_indices)} held out entirely (no unlabeled/'
          f'reconstruction use either) '
          + (f'-> split same_site_test={len(val_indices)} val / {len(test_indices)} test '
             f'(same sites as train)' if same_site_test else
             f'-> {len(val_indices)} val (real cross-site UDATestDataset used for test)')
          + f'; {len(unlabeled_indices)}/{len(full_train_dataset.unlabeled_indices)} '
            f'unlabeled slices used (max_unlabeled_samples={max_unlabeled_samples})')

    labeled_loader = DataLoader(
        Subset(full_train_dataset, train_labeled_indices),
        batch_size=batch_size, shuffle=True, num_workers=0)
    unlabeled_loader = DataLoader(
        Subset(full_train_dataset, unlabeled_indices),
        batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = None
    if val_indices:
        val_loader = DataLoader(
            _EvalPairs(Subset(full_train_dataset, sorted(val_indices))),
            batch_size=eval_batch_size, shuffle=True, num_workers=0)

    if same_site_test:
        test_loader = None
        if test_indices:
            test_loader = DataLoader(
                _EvalPairs(Subset(full_train_dataset, sorted(test_indices))),
                batch_size=eval_batch_size, shuffle=True, num_workers=0)
    else:
        test_dataset = UDATestDataset(
            root=_resolve(cfg['test_root']),
            train_channels=cfg['train_channels'],
        )
        test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=True, num_workers=0)

    num_classes = len(cfg['train_channels'])   # one class per kept channel
    in_channels = 1   # CardiacUDA is single-channel grayscale ultrasound

    return labeled_loader, unlabeled_loader, val_loader, test_loader, num_classes, in_channels
