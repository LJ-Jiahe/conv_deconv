import torch
import torchvision.datasets as datasets
import torch.nn as nn
import os
import functions
import classes
import config as cfg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import thermal_data_loader


train_dataset = thermal_data_loader.Thermal_RGB(cfg.data_folder)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=cfg.train_batch_size,
    shuffle=True)

for train_ite, train_datapoint in enumerate(train_loader):
    train_input_batch = train_datapoint['input_image'].type(torch.FloatTensor)
    train_target_batch = train_datapoint['output_image'].type(torch.FloatTensor)

    fig = plt.figure(figsize=(18, 9))
    fig.add_subplot(1, 2, 1)
    plt.imshow(train_input_batch[0][0])
    fig.add_subplot(1, 2, 2)
    plt.imshow(train_target_batch[0][0])

plt.show()