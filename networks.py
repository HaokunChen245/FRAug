import torchvision
import torch.nn as nn
import os
import torch
import random
import torch.nn.functional as F
from collections import OrderedDict
import torch.nn as nn
from torch.nn import functional as F

class G(nn.Module):    
    #Feature Generation Network.
    def __init__(self, feature_dim, latent_dim, num_class):
        super(G,self).__init__()
        self.num_class = num_class

        self.decoder = nn.Sequential(OrderedDict({
                f'fc0': nn.Linear(latent_dim + num_class, feature_dim),
                f'local_bn0': nn.BatchNorm1d(feature_dim),
                f'relu0': nn.ReLU(inplace=True),
                
                f'fc1': nn.Linear(feature_dim, feature_dim),
                f'local_bn1': nn.BatchNorm1d(feature_dim),
                f'relu1': nn.ReLU(inplace=True),  
                }))
    
    def forward(self, z, y):    
        y_one_hot = F.one_hot(y.view((y.shape[0],)), num_classes=self.num_class).to(y.device)
        o = torch.cat([z, y_one_hot], 1)
        o = self.decoder(o)
        return o 
                
class FTNet(nn.Module):    
    #Feature Transformation Network.
    def __init__(self, feature_dim, latent_dim, num_class, with_BN=True, with_cls=True):
        super(FTNet,self).__init__()
        self.num_class = num_class
        self.with_cls = with_cls
        if with_cls:
            input_dim = feature_dim + num_class
        else:
            input_dim = feature_dim

        self.encoder = nn.Sequential(OrderedDict({
                f'fc0': nn.Linear(input_dim, latent_dim),
                f'bn0': nn.BatchNorm1d(latent_dim),
                f'relu0': nn.ReLU(inplace=True),
                }))

        if with_BN:
            self.decoder = nn.Sequential(OrderedDict({                
                    f'fc1': nn.Linear(latent_dim, feature_dim),
                    f'bn1': nn.BatchNorm1d(feature_dim), #use BN is better for PACS
                    # f'relu1': nn.ReLU(inplace=True),
            }))
        else:
            self.decoder = nn.Sequential(OrderedDict({                
                    f'fc1': nn.Linear(latent_dim, feature_dim),
                    # f'bn1': nn.BatchNorm1d(feature_dim), #use BN is better for PACS
                    # f'relu1': nn.ReLU(inplace=True),
            }))

    def forward(self, f_fake, f_real, y):
        
        # if self.with_cls:
        y_one_hot = F.one_hot(y.view((y.shape[0],)), num_classes=self.num_class).to(y.device)        
        x = torch.cat([f_fake, y_one_hot], 1)  
        # else:
            # x = f_fake

        o = self.encoder(x)
        o = self.decoder(o)
        return f_real + o       

class CNN_Digits(nn.Module):
    """
    Reference
    https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/modeling/backbone/cnn_digit5_m3sda.py
    """

    def __init__(self, class_num=10):
        super().__init__()
        self.encoder_convs = nn.Sequential(OrderedDict({
            f'conv0': nn.Conv2d(3, 64, 5, 1, 2),
            f'bn0': nn.BatchNorm2d(64),
            f'relu0': nn.ReLU(inplace=True),
            f'pooling0': nn.MaxPool2d(3, 2, 1),

            f'conv1': nn.Conv2d(64, 64, 5, 1, 2),
            f'bn1': nn.BatchNorm2d(64),
            f'relu1': nn.ReLU(inplace=True),
            f'pooling1': nn.MaxPool2d(3, 2, 1),

            f'conv2': nn.Conv2d(64, 128, 5, 1, 2),
            f'bn2': nn.BatchNorm2d(128),
            f'relu2': nn.ReLU(inplace=True),

            f'flatten': nn.Flatten(),
        }))     

        self.encoder_fcs = nn.Sequential(OrderedDict({
            f'fc3': nn.Linear(8192, 2048),
            f'bn3': nn.BatchNorm1d(2048),
            f'relu3': nn.ReLU(inplace=True),

            f'fc4': nn.Linear(2048, 512),
            f'bn4': nn.BatchNorm1d(512),
            f'relu4': nn.ReLU(inplace=True),
        }))     

        self.classifier = nn.Sequential(OrderedDict({
            f'fc5': nn.Linear(512, class_num),
        }))     

        self.feature_dim = 512

    def _check_input(self, x):
        H, W = x.shape[2:]        
        assert (
            H == 32 and W == 32
        ), "Input to network must be 32x32, " "but got {}x{}".format(H, W)

    def forward(self, x):
        self._check_input(x)
        o = self.encoder_convs(x)
        o = self.encoder_fcs(o)
        o = self.classifier(o)
        return o
        
