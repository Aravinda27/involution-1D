#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 09:50:30 2024

@author: user1
"""
from involution_1D import Involution1d
import torch
from torch import nn
import numpy as np
#from typing import Union, Tuple, Optional
#import torch.nn.functional as F

class SpeakerEncoder(nn.Module):
    def __init__(self, device, loss_device):
        super().__init__()
        self.loss_device = loss_device

        # Network definition
        self.fCNN = nn.Sequential(
            Involution1d(1, 2, kernel_size=(7, 1), dilation=(2, 1)),
            nn.SELU(),

            Involution1d(2, 4, kernel_size=(7, 1), dilation=(2, 1)),
            nn.SELU(),

            Involution1d(4, 8, kernel_size=(7, 1), dilation=(3, 1)),
            nn.SELU(),

            Involution1d(8, 16, kernel_size=(7, 1), dilation=(4, 1)),
            nn.SELU(),

            Involution1d(16, 32, kernel_size=(7, 1), dilation=(5, 1)),
            nn.SELU(),

            Involution1d(32, 40, kernel_size=(7, 1), dilation=(5, 1)),
            nn.SELU(),

            Involution1d(40, 40, kernel_size=(34, 1)),
        ).to(device)

    def forward(self, x):
        """
        Forward pass for the SpeakerEncoder model.
        
        :param x: (torch.Tensor) Input tensor of shape [batch size, in channels, height, width]
        :return: (torch.Tensor) Output tensor after processing through fCNN
        """
        print(f"Input shape: {x.shape}")
        # Pass the input through the sequential layers defined in fCNN
        for layer in self.fCNN:
            x = layer(x)
            print(f"Shape after {layer.__class__.__name__}: {x.shape}")
        return x

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpeakerEncoder(device, device)

# Assuming the input npy file is loaded as follows
input_data=torch.randn(374,160)

#input_data = np.load("input.npy")
input_tensor = torch.tensor(input_data).unsqueeze(0).unsqueeze(0).float().to(device)  # Shape: (1, 1, 374, 160)

# Forward pass
output = model(input_tensor)
print(f"Output shape: {output.shape}")
