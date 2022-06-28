import torch.nn as nn

from pytorchvideo.models.accelerator.mobile_cpu.efficient_x3d import EfficientX3d
from torch.hub import load_state_dict_from_url

from movinets import MoViNet
from movinets.config import _C as moviConfig


def create_ex3d(args) -> EfficientX3d:
    """This function returns the EfficientX3d module.

    args.model : model name is efficient_x3d_<size>. We load the specified size
    args.transfer : whether we load the pre-trained version of the parameters
    args.num_classes : the output size
    """
    model_size = 'XS'
    if args.model.lower().endswith("_s"):
        model_size = 'S'

    model = EfficientX3d(expansion=model_size, head_act='identity')

    if args.transfer:
        checkpoint_path = 'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/efficient_x3d_xs_original_form.pyth'
        checkpoint = load_state_dict_from_url(checkpoint_path)
        model.load_state_dict(checkpoint)

    features = model.projection.model.in_features
    model.projection.model = nn.Linear(features, args.num_classes)

    return model  # return the model object without any wrapper class


def create_movinet(args) -> MoViNet:
    """This function returns the MoViNet module.

    args.model : model name is movinet_<size>. We load the specified size
    args.transfer : whether we load the pre-trained version of the parameters
    args.num_classes : the output size
    """
    model_config = moviConfig.MODEL.MoViNetA0
    if args.model.endswith("_a2"):
        model_config = moviConfig.MODEL.MoViNetA2
    elif args.model.lower().endswith("_a1"):
        model_config = moviConfig.MODEL.MoViNetA1

    # we use causal=False as we will infer over short clips independently
    # rather than a whole video
    model = MoViNet(model_config, causal=False, pretrained=args.transfer)
    features = model.classifier[-1].conv_1.conv3d.in_channels
    model.classifier[-1].conv_1.conv3d = nn.Conv3d(features, args.num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1))

    return model


def get_model(args) -> nn.Module:
    """This function is the interface to interact with this mini library. It
    returns a nn.Module according to the parameters in args, args.model
    determines what architecture (family and size) will be used.
    """
    if args.model.lower().startswith("efficient_x3d"):
        return create_ex3d(args)
    elif args.model.lower().startswith("movinet"):
        return create_movinet(args)
    else:
        print("Wrong model name")
        return None
