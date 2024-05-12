import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class DomAdaptResNet18(nn.Module):
    def __init__(self, shaping_module, adapt_layers=["layer2.1.conv2"]):
        super(DomAdaptResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

        self.maps_storing_hooks = []
        self.shaping_hooks = []
        self.activation_maps = []

        self.shaping_module = shaping_module

        self.adapt_layers = adapt_layers

    def forward(self, x):
        return self.resnet(x)

    # To store the activation maps
    def register_map_storing_hooks(self):
        for name, module in self.resnet.named_modules():
            if name in self.adapt_layers:
                self.maps_storing_hooks.append(
                    module.register_forward_hook(self.store_maps)
                )

    def store_maps(self, module, input, output):
        self.activation_maps.append(output.clone().detach())

    # To do the activation shaping
    def adapt_activation(self, model, input, output):
        M = self.activation_maps.pop(0)
        return self.shaping_module.shape_activation(output, M)

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
