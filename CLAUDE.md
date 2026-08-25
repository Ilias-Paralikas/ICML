# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Research code for an ICLR submission on unsupervised/semi-supervised cardiac ultrasound
segmentation. The core idea: an autoencoder whose bottleneck is split into several
independent "vectorizers" (one per class), each decoded separately into its own
reconstruction + segmentation channel, so segmentation emerges from how well each
per-class vector reconstructs the image. There is no build system, package config,
linter, or test suite — this is notebook-driven experimentation (`cardiacUDA/*.ipynb`)
backed by plain Python modules (`model/`, `data_handlers/`, `utils/`).

## Running things

There are no CLI entry points — training happens by running notebook cells top to bottom.

- **Main training loop:** `cardiacUDA/train.ipynb`. It appends the repo root to
  `sys.path`, so it must be run with its working directory at `cardiacUDA/`. It is
  dataset-agnostic — set `DATASET_NAME` (in the "DATASET SELECTION" cell) to any key
  registered in `dataset_registry.DATASET_REGISTRY` (currently `'cardiacUDA'` or
  `'camus'`); everything downstream (loaders, model, losses, eval) is generic and needs
  no further edits to switch datasets. The notebook never special-cases any dataset by
  name or introspects what a dataset did internally — the "Generic data loaders" cell
  just calls `build_dataset(DATASET_NAME, batch_size=..., labeled_fraction=...,
  seed=...)` and gets back `labeled_loader, unlabeled_loader, val_loader, test_loader,
  num_classes` — see the Data section below for how each dataset builds those.
  - Runs are versioned per dataset: `root = cache/model_weights/semisup/<DATASET_NAME>/`,
    each with its own independent `get_version_folder` counter — so e.g. `camus` run `12`
    and `cardiacUDA` run `12` are unrelated, separate folders. Set `LOAD_RUN` to an
    existing run number under that dataset's root to resume it (reloads `config.json` +
    `model.pt`); leave it `None` to start a new versioned run.
  - Dataset-independent experiment knobs (lr, `labeled_fraction`, `bottleneck_dim`,
    `vectorizers_mat_mul`, aug params, loss weights) live in `train_config`;
    dataset-derived values (`num_classes`, `batch_size`, `in_channels`, `input_size`) live
    in `dataset_config`. Both are set near the top of the notebook, and `train_config` is
    serialized to `config.json` next to the checkpoint.
  - `evaluate(loader, desc=..., compute_hd=True)` computes Dice (+ Hausdorff distance,
    unless `compute_hd=False`) over any `(image, label)` loader — it doesn't hardcode which
    dataset/split it's given. **Val vs. test discipline:** the per-epoch call inside the
    training loop evaluates on `val_loader` so test stays completely unseen during
    development; the final evaluation cell runs `evaluate(test_loader)` once, at the end.
    HD's per-sample skimage loop is much slower than Dice — if per-epoch val evaluation
    feels too slow, pass `compute_hd=False` to that call for Dice-only. If the selected
    dataset has no val split (i.e. `val_loader` is `None` from `build_dataset()`), the
    training loop skips per-epoch evaluation entirely rather than falling back to test.
  - `cardiacUDA/old_version.ipynb` is a superseded, CardiacUDA-only training notebook kept
    for reference — it predates the `dataset_registry`/`SemiSupervisedDatasetWrapper`
    refactor and duplicates logic (e.g. its own inline `remap_labels`) that now lives in
    `data_handlers/`. Don't use it as the current-state reference; use `train.ipynb`.
- Ad hoc checks are typically run as `python3 -c '...'` rather than through a test runner
  (there is no test suite to invoke).
- `data_handlers/data/` and any `*cache*` directories are git-ignored — raw datasets and
  model checkpoints are expected to exist locally but are never committed.

## Architecture

**Model** (`model/`): `EncoderDecoder` = `VectorEncoder` + `MaskDecoder`.
- `VectorEncoder` runs a conv downsampling stack, then feeds the flattened bottleneck
  into `number_of_vectors` independent `Vectorizer` heads (one per class/channel being
  trained). Each `Vectorizer` is an MLP; if its `use_matrix_multiplication` flag is set,
  its output is a softmax-free linear combination of `number_of_vectors` learned
  prototype vectors (`self.vectors`) instead of a direct MLP output — this is the
  `vectorizers_mat_mul` config knob, applied per-vectorizer.
- `MaskDecoder` takes the list of per-class vectors, concatenates and reshapes them so
  every vector is decoded independently through the same transposed-conv decoder stack
  (batch and vector dims are folded together, then split back apart). The last decoder
  channel is treated as a segmentation logit, the rest as a (sigmoided) reconstruction —
  see the `x.shape[2] == 2` branch in `MaskDecoder.forward` for the 1-vs-multi-channel
  reconstruction split.
- Training combines the per-class reconstructions using the per-class segmentation
  probabilities as soft masks (`reconstruction_loss` in the training notebook), plus a
  supervised dice/CE loss on labeled samples — this is what ties unsupervised
  reconstruction quality to segmentation quality.
- Conv building blocks (`ConvBlock`, `DownConv`, `UpConv`, `ResidualDoubleConv`) live in
  `model/modules/blocks/`.

**Data** (`data_handlers/`):
- `data_loading/dataset_registry.py` maps a dataset name (`'cardiacUDA'` or `'camus'`) to
  a `build_dataset(batch_size, labeled_fraction, seed, eval_batch_size=16)` function that
  returns fully-built `(labeled_loader, unlabeled_loader, val_loader, test_loader,
  num_classes)` — `val_loader` may be `None` if that dataset has no validation split (or,
  for CardiacUDA, if `labeled_fraction` leaves nothing to hold out — see below).
  **Every dataset-specific concern — including whether/how a semi-supervised split
  happens at all — lives entirely inside that function; the registry and the training
  notebook never see raw datasets or special-case any dataset by name.** This is
  deliberate: it's what lets CardiacUDA skip `SemiSupervisedDatasetWrapper` entirely and
  do something dataset-specific internally (see below) without the notebook needing to
  know or care. `labeled_fraction` has the same meaning for every dataset — the fraction
  of the labeled pool actually used as real-labeled training data — even though what
  happens to the rest of that pool differs per dataset (see below). To add a dataset,
  write a `build_dataset()` following `CardiacUDA/build_dataset.py` (or the simpler
  `Camus/build_dataset.py`) and register it.
- `data_loading/label_utils.py` holds the generic `remap_labels`/`remap_sample`: given a
  raw one-hot label and a `train_channels` list of raw channel indices to keep, it drops
  every other channel and folds it back into a recomputed background channel 0. Shared by
  both datasets — `CardiacUDA/datasets/labels.py` just re-exports it.
- `SemiSupervisedDatasetWrapper` (`data_loading/semi_supervised_dataset_wrapper.py`)
  wraps any dataset exposing `labeled_indices` and, seeded once, keeps only a fixed
  `labeled_fraction` of those labels, zeroing the rest and exposing
  `labeled_indices`/`unlabeled_indices`. It is generic and dataset-agnostic on purpose —
  **only `Camus/build_dataset.py` uses it**; CardiacUDA doesn't (see below).
- `CardiacUDA/build_dataset.py` reads `CardiacUDA/dataset_config.json` (paths, `sites`,
  `train_channels`) and builds all four loaders directly, **without
  `SemiSupervisedDatasetWrapper`**. Raw labels are 5-channel one-hot (channel 0 =
  background, 1-4 = cardiac chambers); `train_channels` (e.g. `[0, 3, 4]`) selects which
  chambers are kept, `num_classes` = `len(train_channels)`.
  **Why no wrapper (read the docstring in that file before touching it):** `labeled_fraction`
  of CardiacUDA's labeled pool (`UDATrainDataset.labeled_indices`, ~735 slices — same
  `max(1, int(pool * labeled_fraction))` floor `SemiSupervisedDatasetWrapper` itself uses,
  same `seed`) is kept as real-labeled training data — so far, identical to Camus. Where
  it diverges: Camus treats the *rest* of the labeled pool as unlabeled reconstruction
  data (via the wrapper); CardiacUDA instead holds the rest out **completely** as val — not
  even used for unlabeled reconstruction — because the wrapper's "everything not labeled"
  bucket is its `unlabeled_indices`, which would put those samples back into training.
  Three disjoint index sets result (labeled / val / natively-unlabeled — `UDATrainDataset`
  already zeroes labels for its own unlabeled slices, so no wrapper-style masking is
  needed for those), each turned into a `DataLoader` directly via `Subset`. `val_loader`
  is `None` if `labeled_fraction` leaves nothing in the pool for val (e.g. `1.0`).
- `Camus/build_dataset.py` reads `Camus/dataset_config.json` and builds `CamusDataset` for
  the `'train'`/`'val'`/`'test'` modes (CAMUS ships its own val split on disk), then wraps
  `train_dataset` in `SemiSupervisedDatasetWrapper` normally — the "no anomaly" path. Raw
  CAMUS labels are one-hot `(4, H, W)`: channel 0 = background, 1 = LV endocardium, 2 =
  myocardium, 3 = left atrium. `train_channels` (e.g. `[0, 1, 3]` to drop myocardium) works
  exactly like CardiacUDA's — same `label_utils.remap_sample`, applied inside
  `CamusDataset.__getitem__`; `num_classes` = `len(train_channels)`. Every CAMUS sample has
  ground truth (no unlabeled slices), so `CamusDataset` always reports `is_labeled=True`;
  the wrapper still creates the semi-supervised split by randomly subsampling
  `labeled_fraction` of those labels, honoring whatever the caller passes in.

**Checkpoints/config** (`utils/file_management/`): `get_version_folder(root)` creates an
auto-incrementing run folder (`root/index.txt` tracks the next index) — this is how
`cache/model_weights/semisup/<n>/` run folders are created. `serialize_config` turns a
config dict (including class/type values like optimizer classes) into JSON-safe data for
saving alongside a checkpoint as `config.json`.

**Augmentations** (`utils/augmentation.py`): `GeometricAug` applies one sampled affine
transform per-sample to an image and its label together (bilinear for the image, nearest
for the label, so label values stay binary); `NoiseAug` perturbs only the model input
(noise/brightness/contrast) so the reconstruction target stays clean. `Augmentations` /
`GroupRandomAffine` are an older batch-of-groups augmentation path, not used by the
current training notebook.
