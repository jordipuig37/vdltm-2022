import torch
import torch.quantization as quantization
import torch.quantization.observer as observer


def get_qconfig(args):
    """This function returns the quantization config as a function of the
    arguments.
    """
    if hasattr(args, "qat_observer") and args.qat_observer == "histogram":
        return quantization.QConfig(activation=observer.HistogramObserver, weight=observer.HistogramObserver.with_args(dtype=torch.qint8))

    else:
        return quantization.get_default_qconfig("qnnpack")
