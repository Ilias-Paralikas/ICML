import cv2
import numpy as np

def crop_and_resize_square(image, target_size=256):
    assert len(image.shape) == 2

    height, width = image.shape
  
    if height > width:
        diff = height - width
        crop_top = diff // 2
        crop_bottom = height - (diff - crop_top)
        image = image[crop_top:crop_bottom, :]
    elif width > height:
        diff = width - height
        crop_left = diff // 2
        crop_right = width - (diff - crop_left)
        image = image[:, crop_left:crop_right]
    
    resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    return resized

def normalize_image(image):
    image = image.astype(np.float32)
    image -= image.min()
    image /= image.max()
    return image

def keep_largest_component(mask):
    """Keep only the largest connected component of a binary mask."""
    num, labels = cv2.connectedComponents(mask.astype(np.uint8))
    if num <= 1:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0  # ignore background label
    largest = int(counts.argmax())
    return labels == largest

def channel_mask(mask,channels=5,keep_largest=True):
    assert len(np.unique(mask)) <= channels
    output =np.zeros((channels,*mask.shape),dtype=bool)
    for i in range(channels):
        m = mask == i
        if keep_largest and m.any():
            m = keep_largest_component(m)
        output[i] = m
    if keep_largest:
        output[0] |= ~output.any(axis=0)  # removed dot pixels become background
    return output