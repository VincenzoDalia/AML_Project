import torch
import torch.backends.mps


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


CONFIG = dotdict({})

if torch.cuda.is_available():
    CONFIG.device = "cuda"
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    CONFIG.device = "mps"
else:
    CONFIG.device = "cpu"

CONFIG.dtype = torch.float32

R18_LAYERS_NAMES = {
    "avgpool": "avgpool",
    "1": "conv1",
    "1.0.1": "layer1.0.conv1",
    "1.0.2": "layer1.0.conv2",
    "1.1.1": "layer1.1.conv1",
    "1.1.2": "layer1.1.conv2",
    "1.0.r": "layer1.0.relu",
    "1.0.bn1": "layer1.0.bn1",
    "1.0.bn2": "layer1.0.bn2",
    "1.1.bn1": "layer1.1.bn1",
    "1.1.bn2": "layer1.1.bn2",
    "1.1.r": "layer1.1.relu",
    "2.0.1": "layer2.0.conv1",
    "2.0.2": "layer2.0.conv2",
    "2.1.1": "layer2.1.conv1",
    "2.1.2": "layer2.1.conv2",
    "2.0.bn1": "layer2.0.bn1",
    "2.0.bn2": "layer2.0.bn2",
    "2.1.bn1": "layer2.1.bn1",
    "2.1.bn2": "layer2.1.bn2",
    "2.0.r": "layer2.0.relu",
    "2.1.r": "layer2.1.relu",
    "3.0.1": "layer3.0.conv1",
    "3.0.2": "layer3.0.conv2",
    "3.1.1": "layer3.1.conv1",
    "3.1.2": "layer3.1.conv2",
    "3.0.bn1": "layer3.0.bn1",
    "3.0.bn2": "layer3.0.bn2",
    "3.1.bn1": "layer3.1.bn1",
    "3.1.bn2": "layer3.1.bn2",
    "3.0.r": "layer3.0.relu",
    "3.1.r": "layer3.1.relu",
    "4.0.1": "layer4.0.conv1",
    "4.0.2": "layer4.0.conv2",
    "4.1.1": "layer4.1.conv1",
    "4.1.2": "layer4.1.conv2",
    "4.0.bn1": "layer4.0.bn1",
    "4.0.bn2": "layer4.0.bn2",
    "4.1.bn1": "layer4.1.bn1",
    "4.1.bn2": "layer4.1.bn2",
    "4.0.r": "layer4.0.relu",
    "4.1.r": "layer4.1.relu",
}


def update_config(args):
    active_layers = []
    for arg_layer in args.layers:
        layer_name = R18_LAYERS_NAMES.get(arg_layer)
        if layer_name:
            active_layers.append(layer_name)
        else:
            print(
                f"Warning: Layer '{arg_layer}' does not match any resnet18 layer, pls refer to --help"
            )
            print(
                f"Use the following pattern: LAYER.LEVEL.CONV_NUM, i.e: 2.0.1 for layer2.0.conv1"
            )
            print(f"Ignoring invalid layer...")

    CONFIG.update(vars(args))
    CONFIG.layers = list(set(active_layers))
