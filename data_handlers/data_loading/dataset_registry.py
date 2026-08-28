"""Maps a dataset name to the function that builds ALL of its DataLoaders.

Every dataset-specific concern — paths, how (or whether) it does a semi-supervised
labeled/unlabeled split, where its val set comes from, batching — lives entirely
inside that dataset's own `build_dataset()`. Callers (the training notebook) only
ever see the four loaders + num_classes this module returns; they never need to
know or special-case how any particular dataset got there. This is what lets e.g.
CardiacUDA skip SemiSupervisedDatasetWrapper entirely and do its own thing
internally (see CardiacUDA/build_dataset.py) without the caller noticing.

To add a new dataset: write a `build_dataset(batch_size, labeled_fraction, seed,
eval_batch_size=16, dataset_config_path=None)` for it (see CardiacUDA/build_dataset.py
or Camus/build_dataset.py for the shape — it must return `labeled_loader,
unlabeled_loader, val_loader, test_loader, num_classes, in_channels`, where
`val_loader` may be `None` if that dataset has no validation split defined, and
`in_channels` is that dataset's image channel count — 1 for grayscale ultrasound,
3 for RGB), give the module a module-level `CONFIG_PATH`, and register both below.
"""
from .CardiacUDA.build_dataset import build_dataset as _build_cardiacuda, CONFIG_PATH as _cardiacuda_config_path
from .Camus.build_dataset import build_dataset as _build_camus, CONFIG_PATH as _camus_config_path
from .CelebAMaskHQ.build_dataset import build_dataset as _build_celebamaskhq, CONFIG_PATH as _celebamaskhq_config_path

DATASET_REGISTRY = {
    'cardiacUDA': _build_cardiacuda,
    'camus': _build_camus,
    'celebamaskhq': _build_celebamaskhq,
}

# The *live* dataset_config.json each dataset normally reads from — exposed so a
# caller (the training notebook) can snapshot it into a run folder without needing to
# know each dataset's package layout, and later point build_dataset() at that
# snapshot (via its dataset_config_path kwarg) to reproduce a run exactly, even if
# the live file has since been edited.
DATASET_CONFIG_PATHS = {
    'cardiacUDA': _cardiacuda_config_path,
    'camus': _camus_config_path,
    'celebamaskhq': _celebamaskhq_config_path,
}


def build_dataset(name, **kwargs):
    """Look up `name` in DATASET_REGISTRY and build all of its DataLoaders.

    kwargs (batch_size, labeled_fraction, seed, dataset_config_path, ...) are passed
    straight through to that dataset's own build_dataset() — see its signature for
    what it actually uses; a dataset with its own internal splitting logic
    (CardiacUDA) may ignore some of them on purpose.

    Returns:
        labeled_loader, unlabeled_loader, val_loader, test_loader, num_classes, in_channels
        (val_loader is None if this dataset has no validation split)
    """
    if name not in DATASET_REGISTRY:
        raise KeyError(f'Unknown dataset {name!r}. Available: {list(DATASET_REGISTRY)}')
    return DATASET_REGISTRY[name](**kwargs)


def get_dataset_config_path(name):
    """Return the live dataset_config.json path for `name`."""
    if name not in DATASET_CONFIG_PATHS:
        raise KeyError(f'Unknown dataset {name!r}. Available: {list(DATASET_CONFIG_PATHS)}')
    return DATASET_CONFIG_PATHS[name]
