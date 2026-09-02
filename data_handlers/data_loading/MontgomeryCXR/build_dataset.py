"""Montgomery County CXR loaders — lung-field segmentation, chest X-ray.

Plain "no anomaly" path (see ../npy_folder_build.py): vanilla train/val/test
NpyFolderDataset from the on-disk split written by
preprocess/preprocess_montgomery.py, then SemiSupervisedDatasetWrapper on train.
Grayscale -> build_dataset() returns in_channels=1. 2 classes (background, lung);
`train_channels` in dataset_config.json can drop to background-only if ever wanted.
"""
from pathlib import Path

from ..npy_folder_build import build_npy_folder_loaders

CONFIG_PATH = Path(__file__).parent / 'dataset_config.json'


def build_dataset(batch_size=32, labeled_fraction=0.1, seed=42, eval_batch_size=16,
                  dataset_config_path=None):
    return build_npy_folder_loaders(CONFIG_PATH, batch_size, labeled_fraction, seed,
                                    eval_batch_size, dataset_config_path)
