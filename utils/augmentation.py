import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F

class GroupRandomAffine(nn.Module):
    def __init__(self, degrees=30, translate=None, scale=None, shear=None, interpolation=F.InterpolationMode.BILINEAR):
        super().__init__()
        self.affine = T.RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=interpolation
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
                 shear=15,
                 add_checkboard= True):
        
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.add_checkboard =add_checkboard

        self.affine_augmentations = GroupRandomAffine(
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear
        )

    def __call__(self, batch,train=True):
        if train:
            batch= self.affine_augmentations(batch)

        if self.add_checkboard:
            b_size, frames,channels, h, w = batch.shape
        
            y = torch.arange(h).unsqueeze(1)
            x = torch.arange(w).unsqueeze(0)

            checkerboard = ((x + y) % 2).float()          # (256,256)
            checkerboard = checkerboard.unsqueeze(0).unsqueeze(0)  # (1,1,256,256)

            checkerboard = checkerboard.repeat(b_size, 1,1, 1, 1)  # (b_size,1,256,256)

            batch = batch +(1-batch.detach())*checkerboard
        return batch 

            