"""Builds all of CelebAMask-HQ's DataLoaders from this dataset's own JSON config
(dataset_config.json, next to this file).

Mirrors Camus/build_dataset.py — the "no anomaly" path: build the vanilla
train/val/test datasets, then hand the train one to the generic
SemiSupervisedDatasetWrapper exactly like Camus does. CelebAMask-HQ is fully
labeled (every sample has a mask) and ships an on-disk train/val/test split
(CelebA's official partition, applied in preprocessing), so there is nothing
dataset-specific for the caller to know about — see dataset_registry.py for why
this function owns the whole loader-building pipeline.
"""
import json
from pathlib import Path

from torch.utils.data import DataLoader, Subset

from .datasets import CelebAMaskHQDataset
from ..semi_supervised_dataset_wrapper import SemiSupervisedDatasetWrapper

CONFIG_PATH = Path(__file__).parent / 'dataset_config.json'


def _resolve(path_str):
    """Paths in dataset_config.json are relative to this file's own directory,
    not to whatever notebook/script happens to call build_dataset() — this
    keeps the config caller-cwd-independent."""
    return str((CONFIG_PATH.parent / path_str).resolve())


def build_dataset(batch_size=32, labeled_fraction=0.1, seed=42, eval_batch_size=16,
                   dataset_config_path=None):
    """Build all of CelebAMask-HQ's DataLoaders.

    Fully labeled + on-disk val/test split, no CardiacUDA-style anomaly, so this is
    the plain path: vanilla datasets, then SemiSupervisedDatasetWrapper on the train
    set to carve out `labeled_fraction` of the labels (the rest become unlabeled
    reconstruction data), same as Camus.

    dataset_config_path : read the dataset config from here instead of the live
                          CONFIG_PATH next to this file. Path resolution for entries
                          inside it (root) is still anchored on CONFIG_PATH's own
                          directory regardless — only which file's *content* gets
                          read changes. Used to reproduce an old run exactly from a
                          saved snapshot; see dataset_registry.get_dataset_config_path
                          and the training notebook's LOAD_RUN handling.

    Returns:
        labeled_loader, unlabeled_loader, val_loader, test_loader, num_classes, in_channels
    """
    with open(dataset_config_path or CONFIG_PATH) as f:
        cfg = json.load(f)

    train_dataset = CelebAMaskHQDataset(
        root=_resolve(cfg['root']),
        mode='train',
        train_channels=cfg['train_channels'],
        return_is_labeled=True,
    )
    val_dataset = CelebAMaskHQDataset(
        root=_resolve(cfg['root']),
        mode='val',
        train_channels=cfg['train_channels'],
        return_is_labeled=False,
    )
    test_dataset = CelebAMaskHQDataset(
        root=_resolve(cfg['root']),
        mode='test',
        train_channels=cfg['train_channels'],
        return_is_labeled=False,
    )
    num_classes = len(cfg['train_channels'])   # one class per kept channel
    # RGB face crops — the only dataset in this repo that isn't single-channel
    # ultrasound. Constant for CelebAMask-HQ (the preprocessed images are always
    # 3-channel); returned so the training notebook can set the model's in_channels
    # per-dataset without special-casing any dataset by name.
    in_channels = 3

    wrapped = SemiSupervisedDatasetWrapper(train_dataset, labeled_fraction=labeled_fraction, seed=seed)
    labeled_loader = DataLoader(
        Subset(wrapped, wrapped.labeled_indices), batch_size=batch_size, shuffle=True, num_workers=0)
    unlabeled_loader = DataLoader(
        Subset(wrapped, wrapped.unlabeled_indices), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=True, num_workers=0)

    return labeled_loader, unlabeled_loader, val_loader, test_loader, num_classes, in_channels
