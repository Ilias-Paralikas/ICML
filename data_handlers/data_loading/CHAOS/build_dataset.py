"""CHAOS-CT loaders — liver segmentation, abdominal CT (grayscale).

CHAOS is 3-D DICOM; preprocess/preprocess_chaos_ct.py slices every volume to
256x256 (abdominal HU window -> [0,1]) and writes one folder per axial slice with
a PATIENT-LEVEL 70/15/15 split (no slice leakage). From there it's the plain
"no anomaly" path (see ../npy_folder_build.py): NpyFolderDataset + on-disk split +
SemiSupervisedDatasetWrapper on train. Grayscale -> in_channels=1. 2 classes
(background, liver). Only the CT modality is wired up; MR (4 organs) is not.
"""
from pathlib import Path

from ..npy_folder_build import build_npy_folder_loaders

CONFIG_PATH = Path(__file__).parent / 'dataset_config.json'


def build_dataset(batch_size=32, labeled_fraction=0.1, seed=42, eval_batch_size=16,
                  dataset_config_path=None):
    return build_npy_folder_loaders(CONFIG_PATH, batch_size, labeled_fraction, seed,
                                    eval_batch_size, dataset_config_path)
