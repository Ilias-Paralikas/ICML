# ssl_baselines — semi-supervised segmentation baselines

Run standard semi-supervised segmentation methods on **the same datasets, splits,
seeds and scoring** as our own method, so the comparison table is apples-to-apples.

Nothing is re-formatted (no h5 / nii.gz). The trainer calls
`dataset_registry.build_dataset()` for the loaders and `utils.evaluation.evaluate`
for the metrics — the exact code paths our method uses.

## Methods

| `--method` | Paper | Notes |
|---|---|---|
| `meanteacher` | Tarvainen & Valpola, NeurIPS 2017 (seg form) | EMA teacher + softmax-MSE consistency. The permanent baseline. |
| `uamt` | Yu et al., MICCAI 2019 | Mean Teacher + MC-dropout uncertainty gating on the consistency loss. U-Net gets dropout 0.5. |
| `bcp` | Bai et al., CVPR 2023 | Bidirectional copy-paste. Phase 1: supervised copy-paste pre-train on labelled data (`pretrain_iters`, default 2000). Phase 2: copy-paste between labelled + unlabelled, EMA teacher gives pseudo-labels on the unlabelled crop. The current "SOTA to beat". |

Reimplemented in a common harness (the official BCP/SSL4MIS code is
ACDC/LA-hardcoded and will not run on ISIC/Drishti/Montgomery). Mechanisms follow
the papers; hyperparameter defaults follow SSL4MIS (`ema_decay 0.99`,
`consistency 0.1`, sigmoid ramp-up 200 steps, `mask_frac 2/3`). Backbone is a
plain 2D U-Net sized from the dataset's `in_channels` / `num_classes`.

## Run

```bash
# from the repo root
python ssl_baselines/train_ssl.py --dataset drishtigs --method bcp \
    --labeled_fraction 0.1 --seed 42 --epochs 120 --batch_size 16

python ssl_baselines/train_ssl.py --dataset montgomery --method meanteacher \
    --labeled_fraction 0.1 --seed 42 --epochs 120 --batch_size 16
```

Use the **same `--dataset --labeled_fraction --seed`** you use for our method.
Sweep `labeled_fraction` (`0.01 0.05 0.1 0.2`) so the comparison covers the regime
— several of these methods are unstable at 1 % (that is a point in our favour, but
report it fairly).

`--smoke` (tiny net, 2 epochs, 3 iters/epoch) is a fast pipeline sanity check.
`--compute_hd` adds Hausdorff to the per-epoch val eval (slow). `--eval_ema`
scores the EMA teacher instead of the student. `--amp` for mixed precision.

## Output

`cache/model_weights/ssl_baselines/<dataset>/<method>/<run>/`:
* `config.json`     — full run config
* `best_model.pt`   — best foreground-mean-Dice checkpoint on val
* `last_model.pt`
* `test_result.json` — per-class + foreground-mean **Dice / IoU-Jaccard / HD** on
  the held-out test split (one final eval, best-val checkpoint)

## Caveats

* `labeled_fraction` floors at 1 sample (via `SemiSupervisedDatasetWrapper`).
  BCP's copy-paste pre-training needs >= 2 labelled images; with 1 it falls back
  to plain supervised pre-training.
* U-Net has skip connections; our `EncoderDecoder` does not — a real
  architectural difference, report the param counts (U-Net ~2M at `base=32`, our
  model ~130M in the runs seen).
* RGB datasets (`isic2017`, `drishtigs`) work directly — `in_channels` flows from
  `build_dataset()` into the U-Net's first conv.

## Adding a method

Add a class to `methods.py` with `step(x_l, y_l, x_u, it) -> {'loss', 'sup', 'cons'}`
and `after_step(it)` / `eval_model()` (subclass `_EMABase` if it uses an EMA
teacher), register it in `METHODS`, and add its defaults to `default_cfg`.
Good next candidates: UniMatch (2D, RGB-friendly), ABD (CVPR 2024).
