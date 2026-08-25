"""Builds all of Camus's DataLoaders from this dataset's own JSON config
(dataset_config.json, next to this file).

Mirrors CardiacUDA/build_dataset.py — see that file for the general shape, and
dataset_registry.py for why this function owns the full loader-building pipeline
instead of just handing back raw datasets.
"""
import json
from pathlib import Path

from torch.utils.data import DataLoader, Subset

from .datasets import CamusDataset
from ..semi_supervised_dataset_wrapper import SemiSupervisedDatasetWrapper

CONFIG_PATH = Path(__file__).parent / 'dataset_config.json'


def _resolve(path_str):
    """Paths in dataset_config.json are relative to this file's own directory,
    not to whatever notebook/script happens to call build_dataset() — this
    keeps the config caller-cwd-independent."""
    return str((CONFIG_PATH.parent / path_str).resolve())


def build_dataset(batch_size=32, labeled_fraction=0.1, seed=42, eval_batch_size=16):
    """Build all of Camus's DataLoaders.

    Camus is fully labeled (every sample has ground truth) and has no CardiacUDA-style
    anomaly to handle, so this is the "normal" path: build the vanilla datasets, then
    hand the train one to the generic SemiSupervisedDatasetWrapper exactly like any
    other dataset would.

    Returns:
        labeled_loader, unlabeled_loader, val_loader, test_loader, num_classes
    """
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)

    train_dataset = CamusDataset(
        root=_resolve(cfg['root']),
        mode='train',
        views=cfg['views'],
        frames=cfg['frames'],
        train_channels=cfg['train_channels'],
        return_is_labeled=True,
    )
    # CAMUS ships its own held-out validation split (preprocessed_data/val/).
    val_dataset = CamusDataset(
        root=_resolve(cfg['root']),
        mode='val',
        views=cfg['views'],
        frames=cfg['frames'],
        train_channels=cfg['train_channels'],
        return_is_labeled=False,
    )
    test_dataset = CamusDataset(
        root=_resolve(cfg['root']),
        mode='test',
        views=cfg['views'],
        frames=cfg['frames'],
        train_channels=cfg['train_channels'],
        return_is_labeled=False,
    )
    num_classes = len(cfg['train_channels'])   # one class per kept channel

    wrapped = SemiSupervisedDatasetWrapper(train_dataset, labeled_fraction=labeled_fraction, seed=seed)
    labeled_loader = DataLoader(
        Subset(wrapped, wrapped.labeled_indices), batch_size=batch_size, shuffle=True, num_workers=0)
    unlabeled_loader = DataLoader(
        Subset(wrapped, wrapped.unlabeled_indices), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=True, num_workers=0)

    return labeled_loader, unlabeled_loader, val_loader, test_loader, num_classes
