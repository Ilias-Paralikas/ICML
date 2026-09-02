"""ISIC 2017 loaders — skin-lesion segmentation, dermoscopy (RGB).

Plain "no anomaly" path (see ../npy_folder_build.py): vanilla train/val/test
NpyFolderDataset from ISIC's own on-disk split (written by
preprocess/preprocess_isic2017.py), then SemiSupervisedDatasetWrapper on train.
RGB -> build_dataset() returns in_channels=3. 2 classes (background, lesion).
"""
from pathlib import Path

from ..npy_folder_build import build_npy_folder_loaders

CONFIG_PATH = Path(__file__).parent / 'dataset_config.json'


def build_dataset(batch_size=32, labeled_fraction=0.1, seed=42, eval_batch_size=16,
                  dataset_config_path=None):
    return build_npy_folder_loaders(CONFIG_PATH, batch_size, labeled_fraction, seed,
                                    eval_batch_size, dataset_config_path)
