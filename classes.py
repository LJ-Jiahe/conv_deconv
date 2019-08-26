import torch
import torch.nn as nn
from torch.autograd import Variable
import config as cfg
from torch.utils.data import Dataset
import os
import functions
from skimage import io


class Downsamp_Upsamp_Net(nn.Module):

    def __init__(self):
        super(Downsamp_Upsamp_Net, self).__init__()
        #Convolution 1
        self.conv1 = nn.Conv2d(
            in_channels=cfg.upsampling[0, 0], 
            out_channels=cfg.upsampling[0, 1], 
            kernel_size=cfg.upsampling[0, 2],
            stride=cfg.upsampling[0, 3],
            padding=cfg.upsampling[0, 4])
        nn.init.xavier_uniform_(self.conv1.weight) #Xaviers Initialisation
        self.ac_fun1 = nn.ReLU()
        #Max Pool 1
        self.maxpool1= nn.MaxPool2d(kernel_size=2,return_indices=True)

        #Convolution 2
        self.conv2 = nn.Conv2d(
            in_channels=cfg.upsampling[1, 0], 
            out_channels=cfg.upsampling[1, 1], 
            kernel_size=cfg.upsampling[1, 2],
            stride=cfg.upsampling[1, 3],
            padding=cfg.upsampling[1, 4])
        nn.init.xavier_uniform_(self.conv2.weight)
        self.ac_fun2 = nn.ReLU()
        #Max Pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,return_indices=True)

        #Convolution 3
        self.conv3 = nn.Conv2d(
            in_channels=cfg.upsampling[2, 0], 
            out_channels=cfg.upsampling[2, 1], 
            kernel_size=cfg.upsampling[2, 2],
            stride=cfg.upsampling[2, 3],
            padding=cfg.upsampling[2, 4])
        nn.init.xavier_uniform_(self.conv3.weight)
        self.ac_fun3 = nn.ReLU()

        #Deconvolution 1
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=cfg.downsampling[0, 0], 
            out_channels=cfg.downsampling[0, 1], 
            kernel_size=cfg.downsampling[0, 2],
            stride=cfg.downsampling[0, 3],
            padding=cfg.downsampling[0, 4])
        nn.init.xavier_uniform_(self.deconv1.weight)
        self.ac_fun4 = nn.ReLU()
        #Max UnPool 1
        self.maxunpool1 = nn.MaxUnpool2d(kernel_size=2)

        #Deconvolution 2
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=cfg.downsampling[1, 0], 
            out_channels=cfg.downsampling[1, 1], 
            kernel_size=cfg.downsampling[1, 2],
            stride=cfg.downsampling[1, 3],
            padding=cfg.downsampling[1, 4])
        nn.init.xavier_uniform_(self.deconv2.weight)
        self.ac_fun5 = nn.ReLU()
        #Max UnPool 2
        self.maxunpool2 = nn.MaxUnpool2d(kernel_size=2)

        #Deconvolution 3
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=cfg.downsampling[2, 0], 
            out_channels=cfg.downsampling[2, 1], 
            kernel_size=cfg.downsampling[2, 2],
            stride=cfg.downsampling[2, 3],
            padding=cfg.downsampling[2, 4])
        nn.init.xavier_uniform_(self.deconv3.weight)
        self.ac_fun6 = nn.ReLU()

    def forward(self,x):
        out = self.conv1(x)
        out = self.ac_fun1(out)
        # size1 = out.size()
        # out, indices1 = self.maxpool1(out)
        out = self.conv2(out)
        out = self.ac_fun2(out)
        # size2 = out.size()
        # out, indices2 = self.maxpool2(out)
        out = self.conv3(out)
        out = self.ac_fun3(out)

        out = self.deconv1(out)
        out = self.ac_fun4(out)
        # out = self.maxunpool1(out,indices2,size2)
        out = self.deconv2(out)
        out = self.ac_fun5(out)
        # out = self.maxunpool2(out,indices1,size1)
        out = self.deconv3(out)
        out = self.ac_fun6(out)
        return(out)


class ImageDataset(Dataset):

    def __init__(self, data_folder, input_dir, output_dir, transform=None):
        self.data_folder = data_folder
        self.input_dir = input_dir
        self.output_dir = output_dir
        # 
        self.input_contents = os.listdir(os.path.join(data_folder, input_dir))
        self.output_contents = os.listdir(os.path.join(data_folder, output_dir))
        self.input_contents.sort(key=lambda x:int(x.split('.')[0]))
        self.output_contents.sort(key=lambda x:int(x.split('.')[0]))
        # self.input_contents.sort()
        # self.output_contents.sort()
        
        self.transform=transform

    def __len__ (self):
        return len(self.input_contents)

    def __getitem__(self,idx):
        
        input_image = io.imread(os.path.join(self.data_folder, self.input_dir, self.input_contents[idx]))
        output_image = io.imread(os.path.join(self.data_folder, self.output_dir, self.output_contents[idx]))

        sample = {'input_image': self.transform(input_image), 'output_image': self.transform(output_image)}

        return sample


class Upsamp_Downsamp_Net(nn.Module):

    def __init__(self):
        super(Upsamp_Downsamp_Net, self).__init__()
        #Up 1
        self.conv1 = nn.Conv2d(
            in_channels=cfg.upsampling[0, 0], 
            out_channels=cfg.upsampling[0, 1], 
            kernel_size=cfg.upsampling[0, 2],
            stride=cfg.upsampling[0, 3])
        nn.init.xavier_uniform_(self.conv1.weight) #Xaviers Initialisation
        self.ac_fun1 = nn.ReLU()
        #Max Pool 1
        self.maxpool1= nn.MaxPool2d(kernel_size=2,return_indices=True)

        #Up 2
        self.conv2 = nn.Conv2d(
            in_channels=cfg.upsampling[1, 0], 
            out_channels=cfg.upsampling[1, 1], 
            kernel_size=cfg.upsampling[1, 2],
            stride=cfg.upsampling[1, 3])
        nn.init.xavier_uniform_(self.conv2.weight)
        self.ac_fun2 = nn.ReLU()
        #Max Pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,return_indices=True)

        #Up 3
        self.conv3 = nn.Conv2d(
            in_channels=cfg.upsampling[2, 0], 
            out_channels=cfg.upsampling[2, 1], 
            kernel_size=cfg.upsampling[2, 2],
            stride=cfg.upsampling[2, 3])
        nn.init.xavier_uniform_(self.conv3.weight)
        self.ac_fun3 = nn.ReLU()

        #Down 1
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=cfg.downsampling[0, 0], 
            out_channels=cfg.downsampling[0, 1], 
            kernel_size=cfg.downsampling[0, 2],
            stride=cfg.downsampling[0, 3])
        nn.init.xavier_uniform_(self.deconv1.weight)
        self.ac_fun4 = nn.ReLU()
        #Max UnPool 1
        self.maxunpool1 = nn.MaxUnpool2d(kernel_size=2)

        #Down 2
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=cfg.downsampling[1, 0], 
            out_channels=cfg.downsampling[1, 1], 
            kernel_size=cfg.downsampling[1, 2],
            stride=cfg.downsampling[1, 3])
        nn.init.xavier_uniform_(self.deconv2.weight)
        self.ac_fun5 = nn.ReLU()
        #Max UnPool 2
        self.maxunpool2 = nn.MaxUnpool2d(kernel_size=2)

        #Down 3
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=cfg.downsampling[2, 0], 
            out_channels=cfg.downsampling[2, 1], 
            kernel_size=cfg.downsampling[2, 2],
            stride=cfg.downsampling[2, 3])
        nn.init.xavier_uniform_(self.deconv3.weight)
        self.ac_fun6 = nn.ReLU()

    def forward(self,x):
        out = self.conv1(x)
        out = self.ac_fun1(out)
        # size1 = out.size()
        # out, indices1 = self.maxpool1(out)
        out = self.conv2(out)
        out = self.ac_fun2(out)
        # size2 = out.size()
        # out, indices2 = self.maxpool2(out)
        out = self.conv3(out)
        out = self.ac_fun3(out)

        out = self.deconv1(out)
        out = self.ac_fun4(out)
        # out = self.maxunpool1(out,indices2,size2)
        out = self.deconv2(out)
        out = self.ac_fun5(out)
        # out = self.maxunpool2(out,indices1,size1)
        out = self.deconv3(out)
        out = self.ac_fun6(out)
        return(out)
