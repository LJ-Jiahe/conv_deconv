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


train_loss_loc = os.path.join(cfg.loss_folder, 'train_loss')
test_loss_loc = os.path.join(cfg.loss_folder, 'test_loss')


with open(train_loss_loc, 'ab') as fp:
    for i in range(10):
        test = [i, i ** 2]
        pickle.dump(test, fp)

with open(test_loss_loc, 'ab') as fp:
    for i in range(10):
        test = [i, i ** 3]
        pickle.dump(test, fp)




