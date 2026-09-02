"""Shared ``build_dataset()`` body for the plain per-sample-folder datasets.

ISIC 2017 / Montgomery / CHAOS-CT / Drishti-GS are all the "no anomaly" path —
structurally identical to Camus: build vanilla train/val/test ``NpyFolderDataset``s
from an on-disk split, then wrap the train set in ``SemiSupervisedDatasetWrapper``
to carve out ``labeled_fraction`` of the labels. Each dataset's ``build_dataset.py``
is a one-line call into here; the only thing that varies is which
``dataset_config.json`` it points at (``root`` + ``train_channels``).
"""
import json
from pathlib import Path

from torch.utils.data import DataLoader, Subset

from .npy_folder_dataset import NpyFolderDataset
from .semi_supervised_dataset_wrapper import SemiSupervisedDatasetWrapper


def build_npy_folder_loaders(config_path, batch_size, labeled_fraction, seed,
                             eval_batch_size, dataset_config_path):
    """Return (labeled_loader, unlabeled_loader, val_loader, test_loader,
    num_classes, in_channels).

    config_path          : this dataset's live dataset_config.json (a Path).
    dataset_config_path  : if given, read config from here instead (run snapshot);
                           `root` is still resolved relative to config_path's dir.
    val_loader / test_loader are None if that split folder is absent.
    """
    config_path = Path(config_path)
    with open(dataset_config_path or config_path) as f:
        cfg = json.load(f)

    def resolve(p):
        return str((config_path.parent / p).resolve())

    root = resolve(cfg['root'])
    train_channels = cfg['train_channels']

    def make(mode, role_labeled):
        if not Path(root, mode).is_dir():
            return None
        return NpyFolderDataset(root=root, mode=mode, train_channels=train_channels,
                                return_is_labeled=role_labeled)

    train_dataset = make('train', True)
    val_dataset = make('val', False)
    test_dataset = make('test', False)

    num_classes = len(train_channels)
    in_channels = train_dataset.in_channels

    wrapped = SemiSupervisedDatasetWrapper(train_dataset, labeled_fraction=labeled_fraction,
                                           seed=seed)
    # At labeled_fraction == 1.0 every sample is labeled and unlabeled_indices is empty;
    # DataLoader can't wrap an empty Subset. Fall back to reconstructing over the whole
    # train set (the recon path ignores labels anyway) so a fully-supervised run works.
    unlab_indices = wrapped.unlabeled_indices or list(range(len(wrapped)))
    labeled_loader = DataLoader(Subset(wrapped, wrapped.labeled_indices),
                                batch_size=batch_size, shuffle=True, num_workers=0)
    unlabeled_loader = DataLoader(Subset(wrapped, unlab_indices),
                                  batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = (DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=True,
                             num_workers=0) if val_dataset is not None else None)
    test_loader = (DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=True,
                              num_workers=0) if test_dataset is not None else None)

    return labeled_loader, unlabeled_loader, val_loader, test_loader, num_classes, in_channels
