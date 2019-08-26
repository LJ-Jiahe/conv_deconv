import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import config as cfg
import functions

train_loss = []
test_loss = []

train_loss_loc = os.path.join(cfg.loss_folder, 'train_loss')
test_loss_loc = os.path.join(cfg.loss_folder, 'test_loss')


for item in functions.read_from_pickle_file(train_loss_loc):
    train_loss.append(item)

for item in functions.read_from_pickle_file(test_loss_loc):
    test_loss.append(item)

train_loss = np.array(train_loss)
test_loss = np.array(test_loss)

plt.plot(train_loss[0:-1, 0],train_loss[0:-1, 1],label="Training Loss")
plt.plot(test_loss[0:-1, 0],test_loss[0:-1, 1],label="Testing Loss")
plt.ylabel("Loss")
plt.xlabel("iterations")
plt.legend(loc='upper left')
plt.show()
