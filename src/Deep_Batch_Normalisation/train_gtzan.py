import sys
sys.path.append('..')
import torch.backends.cudnn
from CNN import CNN
from torch.utils.tensorboard import SummaryWriter 
from logs import get_summary_writer_log_dir
from Trainer import Trainer
torch.backends.cudnn.benchmark = True
from dataset import getDataLoaders

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


EPOCHS = 200

def main():
    model = CNN(height=80, width=80, channels=1, class_count=10)
    log_dir = get_summary_writer_log_dir('Deep_Batch_Normalisation_',EPOCHS)
    summary_writer = SummaryWriter(str(log_dir),flush_secs=5)
    train_loader, test_loader = getDataLoaders('/user/home/nj18503/Applied-Deep-Learning/data',16)
    trainer = Trainer(model, train_loader, test_loader, summary_writer, DEVICE)
    trainer.train(EPOCHS,2,print_frequency=10,log_frequency=10)
    torch.save({
    'args': ['deep_batch',EPOCHS],
    'model': model.state_dict(),
},  "".join(['deep_batch',str(EPOCHS)]) )
    summary_writer.close()

if __name__ == "__main__":
    main()
