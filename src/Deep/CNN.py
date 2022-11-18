import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn import Flatten as Flatten
from torch.optim.optimizer import Optimizer

class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int):
        super().__init__()
        self.dropout = nn.Dropout(0.25)
        self.class_count = class_count

        self.conv1_left = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(10, 23),
            padding='same'
        )

        self.initialise_layer(self.conv1_left)
        self.pool1_left = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv1_right = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(21, 10),
            padding='same'
        )
        self.initialise_layer(self.conv1_right)
        self.pool1_right = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2_left = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(5, 11),
            padding='same'
        )

        self.initialise_layer(self.conv2_left)
        self.pool2_left = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2_right = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(1, 5),
            padding='same'
        )
        self.initialise_layer(self.conv2_right)
        self.pool2_right = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3_left = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 5),
            padding='same'
        )

        self.initialise_layer(self.conv3_left)
        self.pool3_left = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3_right = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(5, 3),
            padding='same'
        )
        self.initialise_layer(self.conv3_right)
        self.pool3_right = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4_left = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(1, 5),
            padding='same'
        )

        self.initialise_layer(self.conv4_left)
        self.pool4_left = nn.MaxPool2d(kernel_size=(1, 5))

        self.conv4_right = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(5, 1),
            padding='same'
        )
        self.initialise_layer(self.conv4_right)
        self.pool4_right = nn.MaxPool2d(kernel_size=(5, 1))


        self.fc1 = nn.Linear(10240, 200)
        self.initialise_layer(self.fc1)

        self.fc2 = nn.Linear(200, 10)
        self.initialise_layer(self.fc2)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        right_x = F.leaky_relu(self.conv1_right(wav), 0.3)
        right_x = self.pool1_right(right_x)

        right_x = F.leaky_relu(self.conv2_right(right_x), 0.3)
        right_x = self.pool2_right(right_x)

        right_x = F.leaky_relu(self.conv3_right(right_x), 0.3)
        right_x = self.pool3_right(right_x)

        right_x = F.leaky_relu(self.conv4_right(right_x), 0.3)
        right_x = self.pool4_right(right_x)



        left_x = F.leaky_relu(self.conv1_left(wav), 0.3)
        left_x = self.pool1_left(left_x)

        left_x = F.leaky_relu(self.conv2_left(left_x), 0.3)
        left_x = self.pool1_left(left_x)

        left_x = F.leaky_relu(self.conv3_left(left_x), 0.3)
        left_x = self.pool3_left(left_x)

        left_x = F.leaky_relu(self.conv4_left(left_x), 0.3)
        left_x = self.pool4_left(left_x)

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