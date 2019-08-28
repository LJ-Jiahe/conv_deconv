import cv2

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import io


def grayscale_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def scale_rgb(image):

    ifov = [(56 / 4000) , (45 / 3000)]      # degrees per pixel - 56 degrees / 4000 pixel ..
    fov  = [32 / ifov[0], 26 / ifov[1]]
    rgb_pix = [int(26 / ifov[1] / 2), int(32 / ifov[0] / 2)]        # get the right number of pixels

    center_pix = [int(np.ceil(image.shape[0] / 2)), int(np.ceil(image.shape[1] / 2))]

    image = image[center_pix[0] - rgb_pix[0] - 1:center_pix[0] + rgb_pix[0], center_pix[1] - rgb_pix[1] - 1: center_pix[1] + rgb_pix[1]]

    image = grayscale_image(image)

    # resize the image
    scaled_image = cv2.resize(image, (640, 512))

    return scaled_image


class Thermal_RGB(Dataset):
    """Thermal is the input; RGB is the label."""

    def __init__(self, root_dir, transform=None):
        """.
        """
        self.root_dir = root_dir
        thermal_dir   = os.listdir(os.path.join(root_dir, 'thermal_1'))
        rgb_dir = os.listdir(os.path.join(root_dir, 'rgb_1'))
        self.images = [name.strip('.TIFF') for name in thermal_dir]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):


        thermal_img = cv2.imread(os.path.join(os.path.join(self.root_dir, 'thermal'), self.images[idx]+'.TIFF'), -1)
        rgb_img = cv2.imread(os.path.join(os.path.join(self.root_dir, 'rgb'), self.images[idx]+'_8b.JPG'))
        rgb_img = scale_rgb(rgb_img)
        thermal_img, rgb_img = thermal_img / (2**14), rgb_img / 255

        sample = {'input_image': thermal_img[np.newaxis, ...], 'output_image': thermal_img[np.newaxis, ...]}

        return sample

if __name__ == '__main__':

    root_directory = '/Users/brycemurray/PycharmProjects/thermal_trans/Test_Images'
    data = Thermal_RGB(root_directory)

    for da, lab in data:
        plt.matshow(da)
        plt.figure()
        plt.matshow(lab)
        plt.show()

    print('Finished')

