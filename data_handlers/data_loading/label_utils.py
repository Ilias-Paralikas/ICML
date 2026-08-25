"""Generic one-hot label remapping, shared by every dataset's `Dataset` classes.

Labels are stored as binary one-hot tensors of shape (C, H, W):
    channel 0 : background
    channels 1+ : foreground classes (chambers, structures, ...)

Training often keeps only a subset of these channels (``train_channels``).
Every channel NOT in ``train_channels`` is folded into the background, and the
reduced label is re-expressed in a fresh one-hot space where channel 0 is always
the background. Used by both CardiacUDA (`CardiacUDA/datasets/labels.py`
re-exports these) and Camus.
"""
import torch


def remap_labels(labels, train_channels):
    """Keep only the channels listed in train_channels.

    The first channel of the result is the background: it is recomputed as the
    complement of the union of all other kept channels, so every channel NOT in
    train_channels becomes part of the background.

    labels         : (B, C, H, W) binary one-hot
    train_channels : raw channel indices to keep, e.g. [0, 3, 4]
    Returns        : (B, len(train_channels), H, W) binary one-hot
    """
    nc = len(train_channels)
    new = torch.zeros(labels.shape[0], nc, *labels.shape[2:],
                      dtype=labels.dtype, device=labels.device)
    for i, c in enumerate(train_channels):
        if i > 0:
            new[:, i] = labels[:, c]
    new[:, 0] = 1.0 - new[:, 1:].sum(dim=1).clamp(0, 1)
    return new


def remap_sample(label, train_channels):
    """Single-sample version of remap_labels.

    label : (C, H, W) binary one-hot
    Returns : (len(train_channels), H, W) binary one-hot
    """
    return remap_labels(label.unsqueeze(0), train_channels).squeeze(0)
