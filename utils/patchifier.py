import matplotlib.pyplot as plt
import numpy as np
import torch
class Patchifier:
    def __init__(self,keep_patches=None,total_patches=256,patch_size=16):
        
        self.total_patches =total_patches
        self.patch_size = patch_size

        # if not specified, keep all patches
        if keep_patches is None:
            self.keep_patches = list(range(total_patches))
            self.removed_patches = []
        else:
            self.keep_patches = keep_patches
            self.removed_patches = list(set(range(total_patches))-set(keep_patches))
    
    def patchify_batch(self,image):

        assert len(image.shape) == 4, "Input image must be a 4D tensor (batch_size, channels, height, width)"
        batch_size, channels, height, width = image.shape
        assert height % self.patch_size == 0 and width % self.patch_size == 0

        number_of_patches = (height // self.patch_size) * (width // self.patch_size)
        patches = image.unfold(2, self.patch_size, self.patch_size)
        patches = patches.unfold(3, self.patch_size, self.patch_size)

        patches = patches.permute(0, 2, 3, 1, 5, 4)

        patches = patches.reshape(batch_size, number_of_patches, channels, self.patch_size, self.patch_size)

        return patches
    
    def unpatchify_batch(self,patches):
      
        B, N, C, H, W = patches.shape
        grid_size = int(N ** 0.5)  # Should be 16

        patches = patches.reshape(B, grid_size, grid_size, C, H, W)  # (B, 16, 16, C, 16, 16)

        # Step 2: Undo the earlier permute:
        # You originally did: permute(0, 2, 3, 1, 5, 4)
        # So now: reverse that → permute(0, 3, 1, 5, 2, 4)
        patches = patches.permute(0, 3, 1, 5, 2, 4)  # (B, C, 16, 16, 16, 16)

        # Step 3: Combine patch grid and sizes
        reconstructed = patches.reshape(B, C, grid_size * H, grid_size * W)  # (B, C, 256, 256)
        return reconstructed


    
    def reconstruct_image(self,x):
        ordering = self.keep_patches+self.removed_patches
        patches,channels,patch_size,_ = x.shape
        padding = torch.ones((self.total_patches-patches,channels,patch_size,patch_size))

        x = torch.cat((x,padding),dim=0)
        ordering  =torch.tensor(ordering)
        inverse_ordering = torch.argsort(ordering)
        x_reordered = x[ inverse_ordering, :, :, :]
        return x_reordered

def show_patches(img):
    number_of_patches, channels, patch_size, _ = img.shape
    height =int((number_of_patches)**0.5)
    fig, ax = plt.subplots(height,height)
    vmin = 0
    vmax = 1
    for j in range(height):
        for i in range(height):
            ax[i,j].imshow(img[i+height*j][0], vmin=vmin, vmax=vmax )
            ax[i,j].axis('off')
    plt.show()
    
