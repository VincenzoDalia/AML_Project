import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from globals import CONFIG


class DAResNet18(nn.Module):
    def __init__(self, shaping_module, adapt_layers=["layer2.1.conv2"]):
        super(DAResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

        self.maps_storing_hooks = []
        self.shaping_hooks = []
        self.activation_maps = []

        self.shaping_module = shaping_module
        self.adapt_layers = adapt_layers

        self.visualize = False
        self.to_visualize = []

    def forward(self, x):
        return self.resnet(x)

    def visualize(self):
        self.eval()
        self.visualize = True

    def store_activation_maps(self, targ_x):
        with torch.autocast(
            device_type=CONFIG.device, dtype=torch.float16, enabled=True
        ):

            self.register_map_storing_hooks()
            # We use torch.no_grad() to avoid computing gradients for
            # the target domain because we are not training on it.
            # We only use it to compute the activation maps for the target domain
            with torch.no_grad():
                self(targ_x)
            self.remove_maps_storing_hooks()

    # To store the activation maps
    def register_map_storing_hooks(self):
        for name, module in self.resnet.named_modules():
            if name in self.adapt_layers:
                self.maps_storing_hooks.append(
                    module.register_forward_hook(self.store_map)
                )

    def store_map(self, module, input, output):
        self.activation_maps.append(output.clone().detach())

    # To do the activation shaping
    def adapt_activation(self, model, input, output):
        M = self.activation_maps.pop(0)
        res = self.shaping_module.shape_activation(output, M)

        if self.visualize:
            self.to_visualize.append(
                {
                    "source": output.clone().detach(),
                    "target": M.clone().detach(),
                    "res": res.clone().detach(),
                }
            )

        return res

    def register_shaping_hooks(self):
        for name, module in self.resnet.named_modules():
            if name in self.adapt_layers:
                self.shaping_hooks.append(
                    module.register_forward_hook(self.adapt_activation)
                )

    def remove_maps_storing_hooks(self):
        for h in self.maps_storing_hooks:
            h.remove()

    def remove_shaping_hooks(self):
        for h in self.shaping_hooks:
            h.remove()
