"""Dataset-agnostic segmentation evaluation over an (image, label) loader.

Extracted from train/train.ipynb. The notebook's ``evaluate(...)`` is now a thin
wrapper around this: it passes ``model`` / ``device`` / ``num_classes`` explicitly
(instead of closing over notebook globals) and, for val-role calls, layers its own
best-checkpoint bookkeeping on top of the returned tensors. Model selection /
checkpoint saving is deliberately NOT done here — this function only measures.
"""
import torch
from tqdm import tqdm

from .metrics import (
    compute_class_dice,
    compute_class_iou,
    compute_class_hd,
    print_dice_report,
    print_iou_report,
    print_hd_report,
)


def evaluate(model, loader, num_classes, device, desc='Evaluating', compute_hd=True):
    """Run Dice + IoU (+ optional Hausdorff distance) over every batch in ``loader``.

    model      : the segmentation model; set to eval() here, caller restores train() if needed.
    loader     : any DataLoader yielding (image, label) batches — val, test, or any
                 other eval-role loader; this function doesn't care which dataset it
                 came from.
    num_classes: number of classes in the model's output space (channel 0 = background).
    device     : where to move each image batch.
    compute_hd : Hausdorff distance is an unbatched, per-sample/per-class skimage call —
                 much slower than the vectorized Dice/IoU. Set False to skip it (e.g. a
                 quick per-epoch check) and get only Dice + IoU back.

    Dice and IoU (Jaccard) are two views of the same overlap; the foreground-mean IoU
    is the mIoU used across the face-parsing / semi-supervised-segmentation literature,
    so it's reported alongside Dice for direct comparison.

    Returns: (class_dice, class_iou, class_hd) — (num_classes,) tensors. class_hd is
             None if compute_hd=False.
    """
    model.eval()
    class_dice = torch.zeros(num_classes)
    class_iou  = torch.zeros(num_classes)
    class_hd   = torch.zeros(num_classes)
    n_batches  = 0

    with torch.no_grad():
        for image, label in tqdm(loader, desc=desc):
            image = image.to(device)
            target = label.argmax(dim=1).cpu()                       # (B, H, W) class indices

            _, seg = model(image)
            pred = seg.argmax(dim=1).cpu()                           # (B, H, W)

            class_dice += compute_class_dice(pred, target, num_classes)
            class_iou  += compute_class_iou(pred, target, num_classes)
            if compute_hd:
                class_hd += compute_class_hd(pred, target, num_classes)
            n_batches += 1

    class_dice /= n_batches
    class_iou  /= n_batches
    print_dice_report(class_dice, num_classes)
    print()
    print_iou_report(class_iou, num_classes)

    if compute_hd:
        class_hd /= n_batches
        print_hd_report(class_hd, num_classes)
    else:
        class_hd = None

    return class_dice, class_iou, class_hd
