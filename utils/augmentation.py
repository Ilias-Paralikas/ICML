import random
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F


class GeometricAug:
    """
    Random affine (rotation, translation, scale, shear) applied identically
    to an image and its label mask.  Labels use nearest-neighbour interpolation
    to keep binary values exact.
    """
    def __init__(self, degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10):
        self.degrees   = degrees
        self.translate = translate
        self.scale     = scale
        self.shear     = shear

    def _sample_params(self, H, W):
        angle = random.uniform(-self.degrees, self.degrees)
        tx    = random.uniform(-self.translate[0] * W, self.translate[0] * W)
        ty    = random.uniform(-self.translate[1] * H, self.translate[1] * H)
        scale = random.uniform(self.scale[0], self.scale[1])
        shear = random.uniform(-self.shear, self.shear)
        return angle, [tx, ty], scale, shear

    def __call__(self, images, labels=None):
        """
        images : (B, C, H, W)
        labels : (B, C, H, W) binary float, or None
        Returns augmented (images,) or (images, labels) — same random params per sample.
        """
        B, _, H, W = images.shape
        aug_imgs  = []
        aug_lbls  = [] if labels is not None else None

        for i in range(B):
            angle, translate, scale, shear = self._sample_params(H, W)
            aug_imgs.append(F.affine(images[i], angle=angle, translate=translate,
                                     scale=scale, shear=shear,
                                     fill=0.0,
                                     interpolation=F.InterpolationMode.BILINEAR))
            if labels is not None:
                lbl = F.affine(labels[i], angle=angle, translate=translate,
                               scale=scale, shear=shear,
                               fill=0.0,
                               interpolation=F.InterpolationMode.NEAREST)
                emerged = (lbl.sum(dim=0) == 0)   # (H, W): filled region is all-zero across classes
                lbl[0] = lbl[0] + emerged.float() # emerged region becomes background (class 0)
                aug_lbls.append(lbl)

        aug_imgs = torch.stack(aug_imgs)
        if labels is not None:
            return aug_imgs, torch.stack(aug_lbls)
        return aug_imgs


class NoiseAug:
    """
    Corrupts an image batch with Gaussian noise and random brightness/contrast.
    Applied to the model INPUT only — the reconstruction target stays clean.
    """
    def __init__(self, noise_std=0.05, brightness=0.2, contrast=0.2):
        self.noise_std  = noise_std
        self.brightness = brightness
        self.contrast   = contrast

    def __call__(self, images):
        """images : (B, C, H, W), assumed in [0, 1]"""
        out = images.clone()

        # Gaussian noise
        if self.noise_std > 0:
            out = out + torch.randn_like(out) * self.noise_std

        # Per-sample brightness and contrast jitter
        B = out.shape[0]
        for i in range(B):
            if self.brightness > 0:
                b = random.uniform(1 - self.brightness, 1 + self.brightness)
                out[i] = out[i] * b
            if self.contrast > 0:
                c      = random.uniform(1 - self.contrast, 1 + self.contrast)
                mean   = out[i].mean()
                out[i] = (out[i] - mean) * c + mean

        return out.clamp(0, 1)

class GroupRandomAffine(nn.Module):
    def __init__(self, degrees=30, translate=None, scale=None, shear=None, interpolation=F.InterpolationMode.BILINEAR):
        super().__init__()
        self.affine = T.RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=interpolation,
            fill=0.0
        )

    def forward(self, batch):
        # batch shape: [B, N, C, H, W]  (N can be ANY size)
        B, N = batch.size(0), batch.size(1)
        out = torch.zeros_like(batch)

        for i in range(B):
            # take first image in the group just to get spatial size for param generation
            ref_img = batch[i, 0]

            # Sample params ONCE per BATCH ELEMENT (shared across all N)
            params = self.affine.get_params(
                self.affine.degrees,
                self.affine.translate,
                self.affine.scale,
                self.affine.shear,
                ref_img.shape[-2:],  # (H, W)
            )

            # Apply SAME params to all images in dim=1
            for j in range(N):
                out[i, j] = F.affine(batch[i, j], *params, interpolation=self.affine.interpolation)

        return out


class Augmentations():
    def __init__(self,
                 degrees=30,
                 translate=(0.1, 0.2),
                 scale=(0.8, 1.2),
                 shear=15):
        
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

        self.affine_augmentations = GroupRandomAffine(
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
        )

    def __call__(self, batch,train=True):
        if train:
            batch= self.affine_augmentations(batch)

      
        return batch 

            