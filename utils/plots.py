import numpy as np
import matplotlib.pyplot as plt

def plot_moving_average(array,window_size =200):
    moving_avg = np.convolve(array, np.ones(window_size)/window_size, mode='valid')
    plt.plot(moving_avg)

def show_pair(x,y):
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(x[0].cpu().numpy())
    axs[1].imshow(y[0].cpu().numpy())
    plt.show()
   

def show_images(images):
    fig, axs = plt.subplots(1,len(images))
    for i,image in enumerate(images):
        axs[i].imshow(image[0].cpu().numpy())
    plt.show()




def show_rgb_images(images):
    fig, axs = plt.subplots(1,len(images))
    for i,image in enumerate(images):
        axs[i].imshow(image[0].cpu().permute(1,2,0).numpy())
    plt.show()