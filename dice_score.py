import torch

def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Dice score on the (H, W) images.

    Args:
        pred, target: torch.Tensor of shape (B, C, H, W) or (H, W).
        smooth: smoothing term to avoid division by zero.

    Returns:
        torch.Tensor of shape (B, C) with the Dice score of each image.
        If inputs are (H, W), a scalar is returned.
    """
    assert pred.shape == target.shape, f"shape mismatch: {pred.shape} vs {target.shape}"
    assert pred.ndim >= 2, "expected at least (H, W) tensors"

    if pred.ndim == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)

    pred = pred.reshape(*pred.shape[:-2], -1)
    target = target.reshape(*target.shape[:-2], -1)

    intersection = (pred * target).sum(-1)
    score = (2 * intersection + smooth) / (pred.sum(-1) + target.sum(-1) + smooth)
    return score


if __name__ == "__main__":
    pred = torch.randn(7, 5, 256, 256).sigmoid().round()
    target = torch.rand(7, 5, 256, 256).round()

    scores = dice_score(pred, target)  # (7, 5)

    print("Per-image Dice scores:\n", scores)
    print("Mean per batch:", scores.mean(dim=1))
    print("Overall mean  :", scores.mean())
