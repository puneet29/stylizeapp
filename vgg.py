from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models


class VGG16(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()

        # Load VGG16 skeleton, pretrained
        vgg16_features = models.vgg16(pretrained=True).features

        # Defining slices
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()

        # Initializing the slices with names sliced acc to paper for relu
        # output
        for i in range(4):
            self.slice1.add_module(str(i), vgg16_features[i])
        for i in range(4, 9):
            self.slice2.add_module(str(i), vgg16_features[i])
        for i in range(9, 16):
            self.slice3.add_module(str(i), vgg16_features[i])
        for i in range(16, 23):
            self.slice4.add_module(str(i), vgg16_features[i])

        # Turn off requires_grad
        if (not requires_grad):
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple(
            'VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return(out)
