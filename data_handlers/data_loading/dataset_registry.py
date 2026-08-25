"""Maps a dataset name to the function that builds ALL of its DataLoaders.

Every dataset-specific concern — paths, how (or whether) it does a semi-supervised
labeled/unlabeled split, where its val set comes from, batching — lives entirely
inside that dataset's own `build_dataset()`. Callers (the training notebook) only
ever see the four loaders + num_classes this module returns; they never need to
know or special-case how any particular dataset got there. This is what lets e.g.
CardiacUDA skip SemiSupervisedDatasetWrapper entirely and do its own thing
internally (see CardiacUDA/build_dataset.py) without the caller noticing.

To add a new dataset: write a `build_dataset(batch_size, labeled_fraction, seed,
eval_batch_size=16)` for it (see CardiacUDA/build_dataset.py or
Camus/build_dataset.py for the shape — it must return `labeled_loader,
unlabeled_loader, val_loader, test_loader, num_classes`, where `val_loader` may be
`None` if that dataset has no validation split defined) and register it below.
"""
from .CardiacUDA.build_dataset import build_dataset as _build_cardiacuda
from .Camus.build_dataset import build_dataset as _build_camus

DATASET_REGISTRY = {
    'cardiacUDA': _build_cardiacuda,
    'camus': _build_camus,
}


def build_dataset(name, **kwargs):
    """Look up `name` in DATASET_REGISTRY and build all of its DataLoaders.

    kwargs (batch_size, labeled_fraction, seed, ...) are passed straight through to
    that dataset's own build_dataset() — see its signature for what it actually uses;
    a dataset with its own internal splitting logic (CardiacUDA) may ignore some of
    them on purpose.

    Returns:
        labeled_loader, unlabeled_loader, val_loader, test_loader, num_classes
        (val_loader is None if this dataset has no validation split)
    """
    if name not in DATASET_REGISTRY:
        raise KeyError(f'Unknown dataset {name!r}. Available: {list(DATASET_REGISTRY)}')
    return DATASET_REGISTRY[name](**kwargs)
