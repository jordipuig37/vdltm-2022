import torch
import torch.nn as nn
from pytorch_lightning.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    QuantizationAwareTraining,
    EarlyStopping,
)

from utils.qconfig import get_qconfig

import torch.quantization as quantization


class QuantStubWrapper(nn.Module):
    """Wrapper class for adding QuantStub/DeQuantStub.
    Source: https://pytorchvideo.org/docs/tutorial_accelerator_use_accelerator_model_zoo#deploy
    """
    def __init__(self, module_in):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = module_in
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


class TestQuantizeCallback(Callback):
    """This class implements a callback on fit end in order to validate and
    test after training with a quantized model.
    """
    def on_fit_end(self, trainer, pl_module):
        pl_module.eval()
        pl_module = QuantStubWrapper(pl_module)
        args = pl_module.hparams
        pl_module.qconfig = get_qconfig(args)
        quantization.prepare(pl_module.model, inplace=True)

        torch.backends.quantized.engine = 'qnnpack'  # fix
        quantization.convert(pl_module, inplace=True)


def prepare_callbacks(args) -> dict:
    """This function prepares the callback list to pass to the Trainer.
    Depending on the parameters given in args some callbacks are added and
    others are not. Base callbacks are ModelCheckpoint (saves the model with
    best validation accuracy) and LearningRateMonitor.

    Returns a dictionary that allow is to access at callback info outside this
    function, this dict needs to be converted to a list before being passed to
    the Trainer.
    """
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        save_top_k=3,
        mode="max",
        dirpath=".ckpts/",  # the checkpoints of this callback and the saved ones are in different folders
        filename=args.run_id + "-{epoch}-{train_acc:.3f}-{val_acc:.3f}"
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks = {"checkpoint_callback": checkpoint_callback, "lr_monitor": lr_monitor}

    if hasattr(args, "quantization_aware_training") and args.quantization_aware_training and hasattr(args, "qat_observer"):
        qat_callback = QuantizationAwareTraining(qconfig=get_qconfig(args), observer_type=args.qat_observer, quantize_on_fit_end=False)
        callbacks["qat_callback"] = qat_callback

    if hasattr(args, "patience"):
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
        )
        callbacks["early_stopping"] = early_stopping

    return callbacks
