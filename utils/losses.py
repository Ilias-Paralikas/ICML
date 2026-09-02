"""Training losses for the split-bottleneck autoencoder.

Extracted verbatim from train/train.ipynb so they can be imported, reused across
notebooks, and unit-tested. All three are pure functions of their tensor
arguments — no notebook/global state.

- ``reconstruction_loss`` ties unsupervised reconstruction quality to
  segmentation: it recombines the per-vectorizer reconstructions using the
  segmentation logits as soft masks and compares to the input.
- ``segmentation_loss`` is the supervised term on labeled samples (foreground CE
  + Dice).
- ``dice_loss`` is the soft multi-class Dice used inside ``segmentation_loss``.
"""
import torch
import torch.nn.functional as F


def reconstruction_loss(rec, image, seg_logits):
    """Combine per-vectorizer reconstructions into one image using the
    segmentation logits as soft masks, then compare with the original image.

    rec        : per-vectorizer reconstruction(s), already sigmoided. MaskDecoder
                 emits (B, N, H, W) for single-channel data (grayscale ultrasound)
                 or (B, N, C, H, W) for multi-channel (RGB CelebAMask-HQ) —
                 whichever matches the model's in_channels.
    seg_logits : (B, N, H, W)  per-vectorizer segmentation logits
    image      : (B, C, H, W)  original input image
    """
    seg_probs = F.softmax(seg_logits, dim=1)   # (B, N, H, W) soft masks over vectorizers
    if rec.dim() == 5:
        # Multi-channel: weight every image channel by the same per-vectorizer mask
        # (unsqueeze the channel axis so seg_probs broadcasts over C).
        combined = (seg_probs.unsqueeze(2) * rec).sum(dim=1)   # (B, C, H, W)
        return F.mse_loss(combined, image)
    combined = (seg_probs * rec).sum(dim=1)    # (B, H, W) final image
    return F.mse_loss(combined, image.squeeze(1))


def dice_loss(probs, target_onehot, eps=1e-6):
    """probs / target_onehot : (B, C, H, W)"""
    B, C = probs.shape[:2]
    p = probs.view(B, C, -1)
    t = target_onehot.view(B, C, -1)
    inter = (p * t).sum(dim=2)
    denom = p.sum(dim=2) + t.sum(dim=2)
    return (1.0 - ((2. * inter + 1e-6) / (denom + 1e-6))).mean()


def segmentation_loss(seg_logits, labels, include_background=True):
    """CE (foreground only) + Dice.

    Labels are expected to be pre-remapped to the model's class space by the
    dataset layer: channel 0 = background, channels 1+ = foreground classes.

    include_background : if True, Dice is averaged over all channels
                          (background included). If False, Dice is restricted
                          to foreground channels only — matching the CE term,
                          which is always foreground-only regardless of this flag.
    """
    target = labels.argmax(dim=1).long()                               # (B, H, W)
    target_oh = F.one_hot(target, num_classes=seg_logits.shape[1])     # (B, H, W, C)
    target_oh = target_oh.permute(0, 3, 1, 2).float()                 # (B, C, H, W)

    # CE on foreground pixels only — flatten spatial dims so mask applies cleanly
    fg_mask = labels[:, 1:].sum(dim=1).bool()                         # (B, H, W)
    logits_flat = seg_logits.permute(0, 2, 3, 1).reshape(-1, seg_logits.shape[1])  # (B*H*W, C)
    target_flat = target.reshape(-1)                                   # (B*H*W,)
    fg_flat     = fg_mask.reshape(-1)                                  # (B*H*W,)
    ce = F.cross_entropy(logits_flat[fg_flat], target_flat[fg_flat])

    # Dice — all channels, or foreground only, depending on include_background
    probs = F.softmax(seg_logits, dim=1)
    if include_background:
        dl = dice_loss(probs, target_oh)
    else:
        dl = dice_loss(probs[:, 1:], target_oh[:, 1:])

    return ce + dl
