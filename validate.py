import os
import torch
import re
import classes
import config as cfg
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import functions
import thermal_data_loader

[model, saved_epoch] = functions.recov_from_ckpt()

# Initialize data loader
#validate_dataset = classes.ImageDataset(
#    data_folder=cfg.data_folder, 
#    input_dir=cfg.validate_input_dir, 
#    output_dir=cfg.validate_output_dir, 
#    transform=cfg.transform)
validate_dataset = thermal_data_loader.Thermal_RGB(cfg.data_folder)

validate_data_loader = torch.utils.data.DataLoader(
    dataset=validate_dataset, 
    batch_size=1, 
    shuffle=False)
                                           
print("\nValidation starts\n")

for ite, datapoint in enumerate(validate_data_loader):
    validate_input_batch = datapoint['input_image'].type(torch.FloatTensor)
    validate_target_batch = datapoint['output_image'].type(torch.FloatTensor)
    if torch.cuda.is_available():
        validate_input_batch = validate_input_batch.cuda()
        validate_target_batch = validate_target_batch.cuda()

    validate_output_batch = model(validate_input_batch)

    validate_input_img = validate_input_batch[0].data.numpy().transpose((1, 2, 0))
    validate_target_img = validate_target_batch[0].data.numpy().transpose((1, 2, 0))
    validate_output_img = validate_output_batch[0].data.numpy().transpose((1, 2, 0))
    imgs = [validate_input_img, validate_target_img, validate_output_img]
    
    fig = plt.figure(figsize=(18, 6))
    columns = 3
    rows = 1
    for i in range(columns * rows):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(imgs[i])
    plt.show()
