"""Segmentation metrics + their pretty-printers.

Extracted verbatim from train/train.ipynb so they can be imported and tested.
All functions are pure; ``compute_class_hd`` is the only one with a non-torch
dependency (``skimage.metrics.hausdorff_distance``, which has no batched form).

- ``dice_score``       : per-sample, per-channel Dice on one-hot / soft masks.
- ``compute_class_*``  : per-class Dice / IoU / Hausdorff for a batch of
                         integer class-index maps (per-sample, then mean over batch).
- ``print_*_report``   : format an accumulated per-class vector for stdout.
"""
import numpy as np
import torch
from skimage.metrics import hausdorff_distance


def dice_score(pred, target, smooth=1e-6):
    """Per-sample, per-channel Dice score — no aggregation across batch or classes.

    pred, target : (B, C, H, W) or (H, W)
    Returns      : (B, C) Dice scores, or a scalar if inputs are (H, W).
    """
    assert pred.shape == target.shape, f"shape mismatch: {pred.shape} vs {target.shape}"
    assert pred.ndim >= 2, "expected at least (H, W) tensors"

    if pred.ndim == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)

    pred = pred.reshape(*pred.shape[:-2], -1)
    target = target.reshape(*target.shape[:-2], -1)

    intersection = (pred * target).sum(-1)
    return (2 * intersection + smooth) / (pred.sum(-1) + target.sum(-1) + smooth)


def compute_class_dice(pred, target, num_classes):
    """Per-class Dice for one batch of predicted/target class-index maps.

    pred, target : (B, H, W) integer class-index tensors
    Returns      : (num_classes,) tensor — Dice per class for this batch
    """
    class_dice = torch.zeros(num_classes)
    for c in range(num_classes):
        p = (pred   == c).float().reshape(pred.shape[0],   -1)
        t = (target == c).float().reshape(target.shape[0], -1)
        inter = (p * t).sum(dim=1)
        denom = p.sum(dim=1) + t.sum(dim=1)
        class_dice[c] = ((2. * inter + 1e-6) / (denom + 1e-6)).mean().item()
    return class_dice


def compute_class_iou(pred, target, num_classes):
    """Per-class IoU (Jaccard) for one batch of predicted/target class-index maps.

    Same shape/aggregation convention as compute_class_dice (per-sample IoU, then
    mean over the batch), so the two are directly comparable. The mean over the
    foreground classes is the mIoU reported across the face-parsing / semi-supervised
    segmentation literature (SemanticGAN, DatasetGAN, HandsOff, ...).

    A class absent from both pred and target in a sample scores 1e-6/1e-6 = 1.0 —
    the same "absent from both -> perfect match" convention compute_class_dice and
    compute_class_hd already use.

    pred, target : (B, H, W) integer class-index tensors
    Returns      : (num_classes,) tensor — IoU per class for this batch
    """
    class_iou = torch.zeros(num_classes)
    for c in range(num_classes):
        p = (pred   == c).float().reshape(pred.shape[0],   -1)
        t = (target == c).float().reshape(target.shape[0], -1)
        inter = (p * t).sum(dim=1)
        union = p.sum(dim=1) + t.sum(dim=1) - inter
        class_iou[c] = ((inter + 1e-6) / (union + 1e-6)).mean().item()
    return class_iou


def compute_class_hd(pred, target, num_classes):
    """Per-class Hausdorff distance for one batch of predicted/target class-index maps.

    pred, target : (B, H, W) integer class-index tensors
    Returns      : (num_classes,) tensor — mean HD per class for this batch, in pixels.

    skimage.metrics.hausdorff_distance has no batched/vectorized form, so this loops
    per-sample, per-class — much slower than compute_class_dice; that's what evaluate()'s
    `compute_hd` flag is for.

    A class absent from both pred and target in a sample scores 0 (perfect match — same
    convention compute_class_dice's epsilon smoothing already uses for the empty/empty
    case). A class present in only one of pred/target has an infinite HD (skimage returns
    inf); rather than let that inf/NaN poison the running average, that sample is instead
    scored as the image diagonal — the largest distance actually possible in the image,
    i.e. the standard "worst case" stand-in for a complete miss.
    """
    diag = float(np.hypot(*pred.shape[-2:]))
    pred_np, target_np = pred.numpy(), target.numpy()

    class_hd = torch.zeros(num_classes)
    for c in range(num_classes):
        total = 0.0
        for b in range(pred_np.shape[0]):
            p, t = pred_np[b] == c, target_np[b] == c
            if not p.any() and not t.any():
                total += 0.0
            elif not p.any() or not t.any():
                total += diag
            else:
                total += hausdorff_distance(p, t)
        class_hd[c] = total / pred_np.shape[0]
    return class_hd


def print_dice_report(class_dice, num_classes):
    """Pretty-print an (accumulated) per-class Dice report from compute_class_dice."""
    print('Dice per class:')
    print(f'  Class 0 (background): {class_dice[0]:.4f}')
    for i in range(1, num_classes):
        print(f'  Class {i}:                {class_dice[i]:.4f}')
    print('\nMean Dice (foreground 1-{}): {:.4f}'.format(num_classes - 1, class_dice[1:].mean()))
    print('Mean Dice (all {}):          {:.4f}'.format(num_classes, class_dice.mean()))


def print_iou_report(class_iou, num_classes):
    """Pretty-print an (accumulated) per-class IoU / Jaccard report from
    compute_class_iou. IoU and the Jaccard index are the same quantity
    (|A n B| / |A u B|); the ISIC / skin-lesion literature calls it Jaccard, most
    other segmentation work calls it (m)IoU. The 'foreground' mean is the one that
    lines up with both."""
    print('IoU / Jaccard per class:')
    print(f'  Class 0 (background): {class_iou[0]:.4f}')
    for i in range(1, num_classes):
        print(f'  Class {i}:                {class_iou[i]:.4f}')
    print('\nMean IoU / Jaccard (foreground 1-{}): {:.4f}'.format(num_classes - 1, class_iou[1:].mean()))
    print('Mean IoU / Jaccard (all {}):          {:.4f}'.format(num_classes, class_iou.mean()))


def print_hd_report(class_hd, num_classes):
    """Pretty-print an (accumulated) per-class Hausdorff-distance report from compute_class_hd."""
    print('Hausdorff distance per class (pixels):')
    print(f'  Class 0 (background): {class_hd[0]:.2f}')
    for i in range(1, num_classes):
        print(f'  Class {i}:                {class_hd[i]:.2f}')
    print('\nMean HD (foreground 1-{}): {:.2f}'.format(num_classes - 1, class_hd[1:].mean()))
    print('Mean HD (all {}):          {:.2f}'.format(num_classes, class_hd.mean()))
