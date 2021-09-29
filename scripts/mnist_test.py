"""Source code to test Pytorch lightning on an MNIST dataset."""
import os
import torch
from torch import Tensor
from absl import logging
from absl import app
from pytorch_lightning import LightningModule, Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.metrics.functional import accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

def _check_cuda_device():
    if not torch.cuda.is_available():
        logging.info("Cuda device is not available")


class MNISTModel(LightningModule):
    def __init__(self, data_dir=PATH_DATASETS, hidden_size=64, learning_rate=2e-4):
        super().__init__()
        self._data_dir = data_dir
        self._learning_rate = learning_rate

        self._dims = (1, 28, 28)
        channels, width, height = self._dims
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Define the torch model
        self._model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=channels * width * height, out_features=hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(in_features=hidden_size, out_features=hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(in_features=hidden_size, out_features=10)
        )

    def forward(self, x: Tensor):
        x = self._model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        predictions = torch.argmax(logits, dim=1)
        acc = accuracy(predictions, y)

        # Log the scalars to Tensorboard
        self.log("val loss", loss, prog_bar=True)
        self.log("val acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr= self._learning_rate)

    def prepare_data(self):
        # Download the data
        MNIST(self._data_dir, train=True, download=True)
        MNIST(self._data_dir, train=False, download=True)

    def setup(self, stage = None):
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self._data_dir, train=True, transform=self._transform)
            self._mnist_train, self._mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == "test" or stage is None:
            self._mnist_test = MNIST(self._data_dir, train=False, transform=self._transform)

    def train_dataloader(self):
        return DataLoader(self._mnist_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self._mnist_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self._mnist_test, batch_size=BATCH_SIZE)

def main(argv):
    del argv
    _check_cuda_device()
    mnist_model =  MNISTModel()
    train_dataset = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    trainer = Trainer(gpus=AVAIL_GPUS,
                      max_epochs=10,
                      progress_bar_refresh_rate=10)

    trainer.fit(mnist_model)

    trainer.test()

if __name__ == "__main__":
    app.run(main)