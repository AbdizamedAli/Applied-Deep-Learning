import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.nn import Flatten as Flatten
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import argparse
from pathlib import Path

torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main():
    train_dataset = GTZAN('/content/drive/MyDrive/Github/train.pkl')
    test_dataset = GTZAN('/content/drive/MyDrive/Github/val.pkl')

    train_loader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=16,
    pin_memory=True,
    num_workers=cpu_count(),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=16,
        num_workers=cpu_count(),
        pin_memory=True,
    )

    model = CNN(height=80, width=80, channels=1, class_count=10)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(lr=0.00005,betas=(0.9,0.999),eps=1e-08)

    log_dir = get_summary_writer_log_dir()
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    trainer = Trainer(
        model, train_loader, test_loader, criterion, optimizer, summary_writer, DEVICE, scheduler
    )

    trainer.train(
        40,
        2,
        print_frequency=10,
        log_frequency=10,
    )

    summary_writer.close()


class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.25)
        self.class_count = class_count

        self.conv1_left = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(10, 23),
            padding = 'same'
        )

        self.conv1_right = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(21, 10),
            padding = 'same'
        )

        self.initialise_layer(self.conv1_left)
        self.pool1_left = nn.MaxPool2d(kernel_size=(2, 2))

        self.initialise_layer(self.conv1_right)
        self.pool1_right = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2_left = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(5, 11),
            padding = 'same'
        )

        self.conv2_right = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(10, 5),
            padding = 'same'
        )

        self.initialise_layer(self.conv2_left)
        self.pool2_left = nn.MaxPool2d(kernel_size=(2, 2))

        self.initialise_layer(self.conv2_right)
        self.pool2_right = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3_left = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 5),
            padding = 'same'
        )

        self.conv3_right = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(5, 3),
            padding = 'same'
        )

        self.initialise_layer(self.conv3_left)
        self.pool3_left = nn.MaxPool2d(kernel_size=(2, 2))

        self.initialise_layer(self.conv3_right)
        self.pool3_right = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4_left = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(2, 4),
            padding = 'same'
        )

        self.conv4_right = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(4, 2),
            padding = 'same'
        )

        self.initialise_layer(self.conv3_left)
        self.pool4_left = nn.MaxPool2d(kernel_size=(1, 5))

        self.initialise_layer(self.conv3_right)
        self.pool4_right = nn.MaxPool2d(kernel_size=(5, 1))

        self.fc1 = nn.Linear(10240,200)
        self.initialise_layer(self.fc1)

        self.fc2 = nn.Linear(200,10)
        self.initialise_layer(self.fc2)


    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        right_x = F.leaky_relu(self.conv1_right(wav),0.3)
        right_x = self.pool1_right(right_x)

        right_x = F.leaky_relu(self.conv2_right(right_x),0.3)
        right_x = self.pool2_right(right_x)

        right_x = F.leaky_relu(self.conv3_right(right_x),0.3)
        right_x = self.pool3_right(right_x)

        right_x = F.leaky_relu(self.conv4_right(right_x),0.3)
        right_x = self.pool4_right(right_x)

        left_x = F.leaky_relu(self.conv1_left(wav),0.3)
        left_x = self.pool1_right(left_x)

        left_x = F.leaky_relu(self.conv2_left(left_x),0.3)
        left_x = self.pool2_right(left_x)

        left_x = F.leaky_relu(self.conv3_left(left_x),0.3)
        left_x = self.pool3_right(left_x)

        left_x = F.leaky_relu(self.conv4_left(left_x),0.3)
        left_x = self.pool4_right(left_x)

        flat = torch.nn.Flatten(1,-1)
        right_x = flat(right_x)
        left_x  = flat(left_x)

        left_right_merged = torch.cat((left_x,right_x),1)
        x = F.leaky_relu(self.fc1(left_right_merged),0.3)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for filename, batch, labels, samples in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()
                logits = self.model.forward(batch)
          
                loss = self.criterion(logits,labels)
                weights = torch.cat([p.view(-1) for n, p in self.model.named_parameters() if '.weight' in n])
                l1_regularization = 0.0001 * torch.norm(weights, 1)
                loss += l1_regularization
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for filename, batch, labels, samples  in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))
        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_loss = total_loss / len(self.val_loader)

        self.summary_writer.add_scalars(
                "accuracy",
                {"test": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")


def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)


def get_summary_writer_log_dir() -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = f'200Epochs_run_'
    
    i = 0
    while i < 1000:
        tb_log_dir = Path("logs") / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    main()