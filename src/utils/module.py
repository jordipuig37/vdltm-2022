import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from utils.custom_models import get_model
from utils.dotdic import DotDic
from utils.qconfig import get_qconfig


class VideoClassifier(pl.LightningModule):
    """This class stores and organizes the code necessary to train and test
    video models for recognizing lane type.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # here we save all arguments in kwargs in self.hparams

        self.model = get_model(DotDic(self.hparams))

        # CUDA for PyTorch
        device = torch.device("cuda" if self.hparams.use_cuda else "cpu")
        print('Using device:', device)
        self.model.to(device)

        self.model.qconfig = get_qconfig(self.hparams)

        # metrics
        self.train_loss = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy(num_classes=self.hparams.num_classes)
        self.fscore = torchmetrics.F1(num_classes=self.hparams.num_classes)
        self.recall = torchmetrics.Recall(num_classes=self.hparams.num_classes)
        self.confmat = torchmetrics.ConfusionMatrix(num_classes=self.hparams.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # The model expects a video tensor of shape (B, C, T, H, W), which is
        # the format provided by the dataset
        x, y = batch["video"], batch["label"]
        prediction = self.model(x)

        # Compute cross entropy loss, loss.backwards will be called behind the scenes
        # by PyTorchLightning after being returned from this method.
        loss = self.train_loss(prediction, y)
        acc = self.train_accuracy(F.softmax(prediction, dim=-1), y)

        # Log the train loss and acc to Tensorboard
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["video"], batch["label"]
        prediction = self.model(x)
        loss = F.cross_entropy(prediction, y)
        acc = self.val_accuracy(F.softmax(prediction, dim=-1), y)

        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["video"], batch["label"]
        prediction = self.model(x)

        loss = F.cross_entropy(prediction, y)
        acc = self.test_accuracy(F.softmax(prediction, dim=-1), y)
        rec = self.recall(F.softmax(prediction, dim=-1), y)
        f1 = self.fscore(F.softmax(prediction, dim=-1), y)
        cm = self.confmat(F.softmax(prediction, dim=-1), y)

        self.log("test_acc", acc, on_epoch=True)
        self.log("test_rec", rec, on_epoch=True)
        self.log("F1_score", f1, on_epoch=True)
        self.log("confusion_matrix", cm, on_epoch=True, reduce_fx=torch.sum)

        return loss

    def configure_optimizers(self):
        """We use Adam optimizer with weight decay and lr_scheduler
        ReduceLROnPlateau with patience=1.
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.1, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
                'strict': True,
                'name': "learning_rate"
            }
        }
