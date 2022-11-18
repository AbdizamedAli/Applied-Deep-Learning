import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn import Flatten as Flatten
from torch.optim.optimizer import Optimizer

class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.1)
        self.class_count = class_count

        self.conv1_left = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(10, 23),
            padding='same'
        )

        self.conv1_right = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(21, 20),
            padding='same'
        )

        self.initialise_layer(self.conv1_left)
        self.pool1_left = nn.MaxPool2d(kernel_size=(1, 20))

        self.initialise_layer(self.conv1_right)
        self.pool1_right = nn.MaxPool2d(kernel_size=(20, 1))

        self.fc1 = nn.Linear(10240, 200)
        self.initialise_layer(self.fc1)

        self.fc2 = nn.Linear(200, 10)
        self.initialise_layer(self.fc2)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        right_x = F.leaky_relu(self.conv1_right(wav), 0.3)
        right_x = self.pool1_right(right_x)

        left_x = F.leaky_relu(self.conv1_left(wav), 0.3)
        left_x = self.pool1_left(left_x)

        flat = torch.nn.Flatten(1, -1)

        right_x = flat(right_x)
        left_x = flat(left_x)

        left_right_merged = torch.cat((left_x, right_x), 1)

        x = F.leaky_relu(self.fc1(left_right_merged), 0.3)

        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)