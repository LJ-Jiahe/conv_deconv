import torch
import torch.nn as nn
from torch.utils.data import random_split
import classes
from torchvision import transforms, utils
from skimage import io, transform
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re
import os
import time
import functions
import config as cfg
import math
from tqdm import tqdm
import pickle
import thermal_data_loader

# All parameters are assigned in "config.py"

# Initialize dataset for train and test data
# train_dataset = classes.ImageDataset(
#     data_folder=cfg.data_folder, 
#     input_dir=cfg.train_input_dir, 
#     output_dir=cfg.train_output_dir, 
#     transform=cfg.transform)

# test_dataset = classes.ImageDataset(
#     data_folder=cfg.data_folder, 
#     input_dir=cfg.test_input_dir, 
#     output_dir=cfg.test_output_dir, 
#     transform=cfg.transform)
train_dataset = thermal_data_loader.Thermal_RGB(cfg.data_folder)
test_dataset = thermal_data_loader.Thermal_RGB(cfg.data_folder)

# Initialize data loaders for train and test dataset
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=cfg.train_batch_size,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=cfg.test_batch_size,
    shuffle=False)

# Initialize model either from checkpoints or create a new model
# If not recovering from checkpoints, set saved_epoch to -1
# Put on cuda if possible
if cfg.recov_from_ckpt:
    [model, saved_epoch_idx] = functions.recov_from_ckpt()
else:
    model = classes.Downsamp_Upsamp_Net()
    saved_epoch_idx = 0
    train_loss_loc = os.path.join(cfg.loss_folder, 'train_loss')
    test_loss_loc = os.path.join(cfg.loss_folder, 'test_loss')
    open(train_loss_loc, 'wb').close()
    open(test_loss_loc, 'wb').close()

if torch.cuda.is_available():
    print("\nCuda available\n")
    model.cuda()

# Other parameters
criterion = cfg.criterion
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
#scheduler = torch.optim.lr_scheduler.StepLR(
#    optimizer, 
#    step_size=cfg.lr_step_size, 
#    gamma=cfg.lr_gamma)

# Start training
print("\nTraining Started!\n")
start_time = time.time()

# Epoch number set after saved epoch, i
for epoch in range(saved_epoch_idx + 1, saved_epoch_idx + 1 + cfg.num_epochs):
    print("\nEPOCH " + str(epoch) + " of " + str(saved_epoch_idx + cfg.num_epochs) + "\n")

    train_loss_total = 0
    for train_ite, train_datapoint in enumerate(tqdm(train_loader, desc='Train')):
        #typecasting to FloatTensor as it is compatible with CUDA
        train_input_batch = train_datapoint['input_image'].type(torch.FloatTensor)
        train_target_batch = train_datapoint['output_image'].type(torch.FloatTensor)

        if torch.cuda.is_available():
            train_input_batch = train_input_batch.cuda()
            train_target_batch = train_target_batch.cuda()

        optimizer.zero_grad()
        train_output_batch = model(train_input_batch)
        train_loss = criterion(train_output_batch, train_target_batch)
        train_loss_total += train_loss.item()
        train_loss.backward()
        optimizer.step()
        
    # # Write average loss value to file once every epoch
    train_loss_loc = os.path.join(cfg.loss_folder, 'train_loss')
    train_loss_avg = train_loss_total / train_loader.__len__()
    functions.append_to_pickle_file(train_loss_loc, [epoch, train_loss_avg])
    
 
#  Test after every epoch
    test_loss_total = 0
    for test_ite, test_datapoint in enumerate(tqdm(test_loader, desc='Test')):
        test_input_batch = test_datapoint['input_image'].type(torch.FloatTensor)
        test_target_batch = test_datapoint['output_image'].type(torch.FloatTensor)
    
        if torch.cuda.is_available():
            test_input_batch = test_input_batch.cuda()
            test_target_batch = test_target_batch.cuda()
        
        test_output_batch = model(test_input_batch)
        test_loss = criterion(test_output_batch, test_target_batch)
        test_loss_total += test_loss.item()

    test_loss_avg = test_loss_total / test_loader.__len__()
    test_loss_loc = os.path.join(cfg.loss_folder, 'test_loss')
    functions.append_to_pickle_file(test_loss_loc, [epoch, test_loss_avg])
    

# Print Loss
    time_since_start = (time.time()-start_time) / 60
    print('\nEpoch: {} \nLoss avg: {} \nTest Loss avg: {} \nTime(mins) {}'.format(
         epoch, train_loss_avg, test_loss_avg, time_since_start))

# Save every 50 epochs
    if epoch % 50 == 0:
        ckpt_folder = os.path.join(cfg.ckpt_folder, 'model_epoch_' + str(epoch) + '.pt')
        torch.save(model, ckpt_folder)
        print("\nmodel saved at epoch : " + str(epoch) + "\n")
    
#    scheduler.step() #Decrease learning rate


