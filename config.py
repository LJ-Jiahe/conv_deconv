from torchvision import transforms
from torch import nn
import numpy as np

# Directories
# data_folder = "data/original"
# train_input_dir = "reduced_input"
# train_output_dir = "reduced_output"
# test_input_dir = "reduced_input"
# test_output_dir = "reduced_output"
# validate_input_dir = "reduced_input"
# validate_output_dir = "reduced_output"

data_folder = "data/"
train_input_dir = "train_input"
train_output_dir = "train_output"
test_input_dir = "test_input"
test_output_dir = "test_output"
validate_input_dir = "test_input"
validate_output_dir = "test_output"


loss_folder = "loss"

# Dataset
transform = transforms.Compose([
    transforms.ToTensor()])
    
num_workers = 4

# Model
recov_from_ckpt = False
ckpt_folder = "checkpoints"

# Training parameters
num_epochs = 10
train_batch_size = 1
test_batch_size = 1
criterion = nn.MSELoss()

# Learning rate related
lr = 0.001
lr_step_size = 30
lr_gamma = 0.1


# Network structure
# in_channels, out_channels, kernel_size, stride, padding
upsampling = np.array([
    [1, 64, 4, 2, 0],
    [64, 128, 5, 2, 0],
    [128, 256, 3, 1, 0]
    # [3, 128, 4, 2, 0],
    # [128, 256, 5, 2, 0],
    # [256, 512, 3, 1, 0],
])

downsampling = np.array([
    [256, 128, 3, 1, 0],
    [128, 64, 5, 2, 0],
    [64, 1, 4, 2, 0]
    # [512, 256, 3, 1, 0],
    # [256, 128, 5, 2, 0],
    # [128, 3, 4, 2, 0]
])

#upsampling = np.array([
#    [3, 16, 4, 2, 0],
#    [16, 32, 5, 2, 0],
#    [32, 64, 3, 1, 0]
#])
#
#downsampling = np.array([
#    [64, 32, 3, 1, 0],
#    [32, 16, 5, 2, 0],
#    [16, 3, 4, 2, 0]
#])
