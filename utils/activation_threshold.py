import torch
def activation_threshold(dataset):
    x =dataset[0]
    mask = torch.zeros_like(x)
    for  x in dataset:
        non_zero = x!=0
        mask[non_zero] +=1
    
    # empirically determined threshold
    binary_mask = mask >(torch.mean(mask)/3)
    # ignore the line showing the pulse
    binary_mask[:,:,:17]=0
    return binary_mask
