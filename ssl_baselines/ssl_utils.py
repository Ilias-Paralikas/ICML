"""Shared pieces for the SSL baselines: EMA teacher, ramp-up schedule, the
supervised CE+Dice loss, consistency loss, MC-dropout uncertainty, and the
bidirectional copy-paste used by BCP.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def ema_update(student, teacher, decay):
    """teacher <- decay * teacher + (1 - decay) * student  (params); buffers copied."""
    d = min(decay, 1.0)
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(d).add_(ps.data, alpha=1.0 - d)
    for bs, bt in zip(student.buffers(), teacher.buffers()):
        bt.data.copy_(bs.data)


def sigmoid_rampup(current, rampup_length):
    """0 -> 1 over `rampup_length` steps, sigmoid shape (Tarvainen & Valpola)."""
    if rampup_length <= 0:
        return 1.0
    x = float(np.clip(current, 0.0, rampup_length))
    phase = 1.0 - x / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def dice_ce_loss(logits, target_onehot, pix_weight=None):
    """Standard supervised term for these baselines: 0.5*CE + 0.5*(1 - soft Dice),
    averaged over ALL classes (matches SSL4MIS / BCP).

    logits        : (B, C, H, W)
    target_onehot : (B, C, H, W) binary
    pix_weight    : optional (B, 1, H, W) per-pixel weight in [0, 1] (BCP uses this
                    to down-weight pseudo-labelled regions).
    """
    B, C = logits.shape[:2]
    tgt_idx = target_onehot.argmax(1)
    if pix_weight is None:
        ce = F.cross_entropy(logits, tgt_idx)
        probs = logits.softmax(1)
        p = probs.reshape(B, C, -1)
        t = target_onehot.reshape(B, C, -1)
        inter = (p * t).sum(-1)
        denom = p.sum(-1) + t.sum(-1)
    else:
        w = pix_weight
        ce_map = F.cross_entropy(logits, tgt_idx, reduction='none').unsqueeze(1)
        ce = (ce_map * w).sum() / (w.sum() + 1e-6)
        probs = logits.softmax(1)
        inter = (probs * target_onehot * w).sum(dim=(2, 3))
        denom = (probs * w).sum(dim=(2, 3)) + (target_onehot * w).sum(dim=(2, 3))
    dice = 1.0 - ((2.0 * inter + 1e-5) / (denom + 1e-5)).mean()
    return 0.5 * ce + 0.5 * dice


def softmax_mse(logits_a, logits_b):
    """Mean-Teacher consistency: MSE between the two softmax outputs."""
    return F.mse_loss(logits_a.softmax(1), logits_b.softmax(1))


def enable_dropout(model):
    """Put ONLY the dropout layers into train mode (BN stays in eval / running
    stats) — the correct way to do MC-dropout with a BatchNorm net."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


@torch.no_grad()
def mc_uncertainty(model, x, passes=8):
    """UA-MT uncertainty: run `model` `passes` times with dropout on, return
    (mean_probs (B,C,H,W), entropy (B,1,H,W)). entropy in [0, ln C]."""
    model.eval()
    enable_dropout(model)
    probs = None
    for _ in range(passes):
        p = model(x).softmax(1)
        probs = p if probs is None else probs + p
    probs = probs / passes
    ent = -(probs * torch.log(probs + 1e-6)).sum(1, keepdim=True)
    return probs, ent


def rand_box_mask(b, h, w, side_frac, device, generator=None):
    """Binary mask (b, 1, h, w): 1 inside a random axis-aligned box whose side is
    `side_frac` of the image (BCP's copy-paste region; paper uses ~2/3)."""
    bh = max(1, int(round(h * side_frac)))
    bw = max(1, int(round(w * side_frac)))
    m = torch.zeros(b, 1, h, w, device=device)
    for i in range(b):
        top = int(torch.randint(0, h - bh + 1, (1,), generator=generator))
        left = int(torch.randint(0, w - bw + 1, (1,), generator=generator))
        m[i, 0, top:top + bh, left:left + bw] = 1.0
    return m


def bcp_mix(x_a, y_a, x_b, y_b, mask):
    """Bidirectional copy-paste. mask == 1 -> take source A inside the box.

    Returns (x_in, y_in, x_out, y_out):
      x_in  = A inside the box, B outside     (labeled centre, unlabeled surround)
      x_out = B inside the box, A outside     (unlabeled centre, labeled surround)
    Labels y_* are mixed the same way (A = GT one-hot, B = pseudo-label one-hot).

    x_*: (B, C, H, W)   y_*: (B, K, H, W)   mask: (B, 1, H, W)
    """
    x_in = x_a * mask + x_b * (1 - mask)
    x_out = x_b * mask + x_a * (1 - mask)
    y_in = y_a * mask + y_b * (1 - mask)
    y_out = y_b * mask + y_a * (1 - mask)
    return x_in, y_in, x_out, y_out


def onehot(idx, num_classes):
    """(B, H, W) int -> (B, C, H, W) float one-hot."""
    return F.one_hot(idx.long(), num_classes).permute(0, 3, 1, 2).float()
