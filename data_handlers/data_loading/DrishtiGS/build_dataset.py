"""Drishti-GS1 loaders — optic disc / cup segmentation, retinal fundus (RGB).

Plain "no anomaly" path (see ../npy_folder_build.py): vanilla train/val/test
NpyFolderDataset from the split written by preprocess/preprocess_drishtigs.py
(Drishti's own Test split + a seeded 40/10 train/val carve of its Training set),
then SemiSupervisedDatasetWrapper on train. RGB -> in_channels=3. 3 nested classes
(background, optic disc rim, optic cup).
"""
from pathlib import Path

from ..npy_folder_build import build_npy_folder_loaders

CONFIG_PATH = Path(__file__).parent / 'dataset_config.json'


def build_dataset(batch_size=32, labeled_fraction=0.1, seed=42, eval_batch_size=16,
                  dataset_config_path=None):
    return build_npy_folder_loaders(CONFIG_PATH, batch_size, labeled_fraction, seed,
                                    eval_batch_size, dataset_config_path)
