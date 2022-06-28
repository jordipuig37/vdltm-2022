import torch
import torch.nn as nn
from pytorchvideo.accelerator.deployment.mobile_cpu.utils.model_conversion import (
    convert_to_deployable_form,
)
from torch.utils.mobile_optimizer import optimize_for_mobile

from utils.module import VideoClassifier
from utils.qconfig import get_qconfig


def load_model(model_args: dict):
    """This function reads the configuration used to to train a model and loads
    the trained version of this model. Now only supporting VideoClassifier."""
    raw_module = VideoClassifier(**model_args)
    trained = raw_module.load_from_checkpoint(model_args.checkpoint_file, strict=False)

    return trained


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


def trace_opt_and_save(input_model, input_tensor, args, strict=False):
    """This function traces the model and saves two scripted versions of it
    one that optimizes for mobile (.ptl) and another one that saves the traced
    model (.pt) that will be used to perform tests in the server.
    """
    traced_model = torch.jit.trace(input_model, input_tensor, strict=strict)
    traced_model.save(args.deployed_server)  # safe .pt

    traced_model_opt = optimize_for_mobile(traced_model)
    traced_model_opt._save_for_lite_interpreter(args.deployed_mobile)  # safe .ptl


def deploy(run_args):
    """This function performs the full process of deploying a model. It loads
    a checkpoint, converts to deployable form, quantizes (if quantize=True) and
    saves two versions: one for mobile and another for testing in the server.
    """
    device = torch.device('cpu')
    model = load_model(run_args).model.to(device)

    input_blob_size = tuple(run_args.input_blob_size)
    input_tensor = torch.randn(input_blob_size, device=device)
    model_to_deploy = convert_to_deployable_form(model, input_tensor)

    if run_args.quantize:
        # quantize the model and save the quantized version
        model_deploy_quant_stub_wrapper = QuantStubWrapper(model_to_deploy)
        model_deploy_quant_stub_wrapper.qconfig = get_qconfig(run_args)
        model_deploy_quant_stub_wrapper_prepared = torch.quantization.prepare(model_deploy_quant_stub_wrapper)

        # calibration is skipped here.
        torch.backends.quantized.engine = 'qnnpack'  # fix for working with mobile
        model_to_deploy = torch.quantization.convert(model_deploy_quant_stub_wrapper_prepared)

    trace_opt_and_save(model_to_deploy, input_tensor, run_args, strict=False)
