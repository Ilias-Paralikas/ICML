"""Training losses for the split-bottleneck autoencoder.

Extracted verbatim from train/train.ipynb so they can be imported, reused across
notebooks, and unit-tested. All three are pure functions of their tensor
arguments — no notebook/global state.

- ``reconstruction_loss`` ties unsupervised reconstruction quality to
  segmentation: it recombines the per-vectorizer reconstructions using the
  segmentation logits as soft masks and compares to the input.
- ``square_then_blend_reconstruction_loss`` is the same idea with the squaring and
  the blending swapped — see its docstring for why that matters.
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


def square_then_blend_reconstruction_loss(rec, image, seg_logits):
    """Per-vectorizer squared error, then blended by the segmentation probabilities.

    Same inputs as ``reconstruction_loss`` — the ONLY difference is the order of the
    squaring and the blending, and that difference is the whole point:

        reconstruction_loss (blend-then-square) :  ( sum_n p_n * r_n  -  x )^2
        this one            (square-then-blend) :  sum_n p_n * ( r_n - x )^2

    Why it matters. Blend-then-square asks "if I MIX the per-class reconstructions in
    ratio p, do I get the image back?" — so p acts as a paint-mixing ratio, and the
    model can score perfectly with two *wrong* templates whose errors cancel. Concrete
    example at one pixel, x = 0.5, r_bg = 0.2, r_liver = 0.8, p = (0.5, 0.5):

        blend-then-square : (0.5*0.2 + 0.5*0.8 - 0.5)^2            = 0.00  <- "perfect"
        square-then-blend : 0.5*(0.2-0.5)^2 + 0.5*(0.8-0.5)^2      = 0.09

    Both templates are wrong by 0.3; only the second says so. Squared errors are
    non-negative, so they cannot cancel — the best the mask can do is put its weight on
    whichever template is *actually* closest. Formally, this term is LINEAR in p, so its
    minimum over the probability simplex is always at a vertex (a hard assignment),
    whereas blend-then-square is quadratic in p and is generally minimised by an
    interior mix. That changes what the segmentation gradient MEANS:

        d/dp_n  square-then-blend  =  (r_n - x)^2         "how badly does template n
                                                           miss this pixel?"  (classify)
        d/dp_n  blend-then-square  =  2(c - x)(r_n - c)   "does more of n push the
                                                           average toward x?"  (shade)

    The blend-then-square gradient also vanishes wherever r_n ~= c, i.e. wherever the
    mask has already committed — which is most of the image — while this one stays
    informative there.

    Scale note: this returns the same order of magnitude as ``reconstruction_loss``
    (both are mean squared errors per pixel, averaged over batch/spatial dims, and over
    channels for multi-channel data), so ``reconstruction_weight`` and the warmup
    schedule stay meaningful when switching between them.

    rec        : per-vectorizer reconstruction(s), already sigmoided. (B, N, H, W) for
                 single-channel data or (B, N, C, H, W) for multi-channel — same
                 convention as ``reconstruction_loss``.
    image      : (B, C, H, W)  original input image
    seg_logits : (B, N, H, W)  per-vectorizer segmentation logits
    """
    seg_probs = F.softmax(seg_logits, dim=1)          # (B, N, H, W) soft masks
    if rec.dim() == 5:
        # Multi-channel: squared error per component, averaged over the image channels
        # so each component still contributes ONE scalar error per pixel (keeps the
        # per-pixel scale identical to the single-channel branch below).
        err = (rec - image.unsqueeze(1)).pow(2).mean(dim=2)   # (B, N, H, W)
    else:
        # Single-channel: image is (B, 1, H, W) and broadcasts against (B, N, H, W).
        err = (rec - image).pow(2)                            # (B, N, H, W)
    return (seg_probs * err).sum(dim=1).mean()


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
