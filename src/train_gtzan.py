import sys
sys.path.append('..')
import torch.backends.cudnn
from Shallow.CNN import CNN
from Deep.DCNN import DCNN
from DeepBn.DCNNBN import DCNNBN
from torch.utils.tensorboard import SummaryWriter 
from Utils.logs import get_summary_writer_log_dir
from Trainer import Trainer
torch.backends.cudnn.benchmark = True
from Utils.dataset import getDataLoaders
from multiprocessing import cpu_count
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Train a CNN model on GTZAN",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--model", default=0, type=int, choices={0,1,2}, help="Choose the model you want to run. For Shallow Network 0. For the Deep Netowork 1.For the Deep Batch Normalisation Network 3")
parser.add_argument(
    "--data-path", 
    default='/user/home/nj18503/Applied-Deep-Learning/data/', 
    type=str, 
    help="The path of where the training and testing dataset are located")
parser.add_argument(
    "--train-data", 
    default='train.pkl', 
    type=str, 
    help="The file name of the training dataset")
parser.add_argument(
    "--test-data", 
    default='val.pkl', 
    type=str, 
    help="The file name of the testing dataset")
parser.add_argument(
    "--batch-size",
    default=128,
    type=int,
    help="Number of audio segments within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=200,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)




if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main(args):
    models = [CNN(),DCNN(),DCNNBN()]
    train_loader, test_loader = getDataLoaders(args.data_path,args.train_data,args.test_data,args.batch_size)
    model = models[args.model]
    log_dir = get_summary_writer_log_dir(args)
    summary_writer = SummaryWriter(str(log_dir),flush_secs=5)
    trainer = Trainer(model, train_loader, test_loader, summary_writer, DEVICE)
    trainer.train(args.epochs,args.val_frequency,print_frequency=args.print_frequency,log_frequency=args.log_frequency)
    summary_writer.close()


if __name__ == "__main__":
    main(parser.parse_args())