import torch

def iou_binary(pred: torch.Tensor, target: torch.Tensor, eps=1e-7):
    """
    pred, target: binary masks of shape (N, H, W) or (H, W)
    """
    pred = pred.bool()
    target = target.bool()

    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()

    return intersection / (union + eps)

import torch

def dice_score(pred, target, eps=1e-6):
    """
    pred   : (B, 1, H, W) or (B, H, W)
    target : same shape as pred
    """
    pred = pred.float()
    target = target.float()
    
    # Flatten
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean()

