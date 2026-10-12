#Import libraries

import torch
from torch import nn


### First Model (CNN)
class PokemonCardModelV1(nn.Module):
    def __init__(self,input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block1= nn.Sequential(
            nn.Conv2d(in_channels = input_shape, out_channels= hidden_units,kernel_size=3,padding=1, stride = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size=3, padding = 1, stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_units, out_channels= hidden_units, kernel_size= 3, padding= 1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels= hidden_units, out_channels= hidden_units, kernel_size= 3, padding=1, stride =1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride =2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=hidden_units, out_features=32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=32, out_features=output_shape)
        )

    def forward(self,x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)


        return x





