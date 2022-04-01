
import numpy as np
from cv2 import imread, imwrite
import os
from glob import glob

def get_aerial_images():
    aerial_images = []
    for parent_directory in glob('/kaggle/input/aerial-tiles-extraction-0-5000/aerials/*'):
        directory_images = sorted(glob(parent_directory + '/*'))
        aerial_images.extend(directory_images)
    return sorted(aerial_images)


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def sample_within_bounds(signal, x, y, bounds):

    xmin, xmax, ymin, ymax = bounds

    idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)
    
    sample = np.zeros((x.shape[0], x.shape[1], signal.shape[-1]))
    sample[idxs, :] = signal[x[idxs], y[idxs], :]

    return sample


def sample_bilinear(signal, rx, ry):

    signal_dim_x = signal.shape[0]
    signal_dim_y = signal.shape[1]

    # obtain four sample coordinates
    ix0 = rx.astype(int)
    iy0 = ry.astype(int)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    bounds = (0, signal_dim_x, 0, signal_dim_y)

    # sample signal at each four positions
    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)

    na = np.newaxis
    # linear interpolation in x-direction
    fx1 = (ix1-rx)[...,na] * signal_00 + (rx-ix0)[...,na] * signal_10
    fx2 = (ix1-rx)[...,na] * signal_01 + (rx-ix0)[...,na] * signal_11

    # linear interpolation in y-direction
    return (iy1 - ry)[...,na] * fx1 + (ry - iy0)[...,na] * fx2


############################ Apply Polar Transform to Aerial Images in CVUSA Dataset #############################
S = 400  # Original size of the aerial image
height = 128  # Height of polar transformed aerial image
width = 512   # Width of polar transformed aerial image

i = np.arange(0, height)
j = np.arange(0, width)
jj, ii = np.meshgrid(j, i)

y = S/2. - S/2./height*(height-1-ii)*np.sin(2*np.pi*jj/width)
x = S/2. + S/2./height*(height-1-ii)*np.cos(2*np.pi*jj/width)

output_dir = '/kaggle/working/polar_aerial_images/'

create_directory(output_dir)

aerial_images = get_aerial_images()

for img in aerial_images[:10000]:
    signal = imread(img)
    image = sample_bilinear(signal, x, y)
    trajectory_dir = output_dir + img[53:-7]
    create_directory(trajectory_dir)
    imwrite(trajectory_dir + img[71:], image)
