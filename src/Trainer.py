import sys
sys.path.append('..')
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import time
from Utils.evalutation import compute_accuracy
import numpy as np
class Trainer:
    """
    A class for training and evaluating a PyTorch model.
    
    Attributes:
        model (nn.Module): The PyTorch model to be trained and evaluated.
        train_loader (DataLoader): A DataLoader for the training dataset.
        val_loader (DataLoader): A DataLoader for the validation dataset.
        summary_writer (SummaryWriter): A TensorBoard SummaryWriter for logging metrics.
        device (torch.device): The device (CPU or GPU) to use for training and evaluation.
        criterion (nn.CrossEntropyLoss): The loss function used for training.
        optimizer (torch.optim.Adam): The optimizer used for training.
        step (int): The current training step.
    
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.00005,betas=(0.9, 0.999), eps=1e-08)
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
        """
        Train the model for the specified number of epochs.
        
        Args:
            epochs (int): The number of epochs to train for.
            val_frequency (int): The frequency (in epochs) at which to run validation.
            print_frequency (int): The frequency (in steps) at which to print training metrics.
            log_frequency (int): The frequency (in steps) at which to log training metrics.
            start_epoch (int): The epoch at which to start training.
        
        """
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for filename, batch, labels, samples in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()
                logits = self.model.forward(batch)

                loss = self.criterion(logits, labels)
                weights = torch.cat(
                    [p.view(-1) for n, p in self.model.named_parameters() if '.weight' in n])
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
                    self.log_metrics(epoch, accuracy, loss,
                                     data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss,
                                       data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        """
        Prints metrics for a given epoch, step, and batch.

        Args:
        epoch (int): current epoch number.
        accuracy (float): batch accuracy.
        loss (float): batch loss.
        data_load_time (float): time to load data for the batch.
        step_time (float): time to run a single step.
        """
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
        """Logs various metrics for a training epoch.

        Args:
        epoch: The number of the epoch being logged.
        accuracy: The accuracy of the model on the training data for the current epoch.
        loss: The loss of the model on the training data for the current epoch.
        data_load_time: The time it took to load the data for the current epoch.
        step_time: The time it took for a single training step in the current epoch.
        """
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

        with torch.no_grad():
            for filename, batch, labels, samples in self.val_loader:
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
        print(
            f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")
