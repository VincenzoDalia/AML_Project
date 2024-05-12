import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# Modifies 'BaseResNet18' including the Activation Shaping Module
class RASResNet18(nn.Module):
    def __init__(
        self, mask_ratio, shaping_module, random_shaping_layers=["layer2.1.conv2"]
    ):
        super(RASResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.shaping_module = shaping_module
        self.random_shaping_layers = random_shaping_layers
        self.hooks = []
        self.mask_ratio = mask_ratio

    def forward(self, x):
        return self.resnet(x)

    def random_shape_activation(self, model, input, output):
        num_elements = output.numel()
        num_zeros = int(num_elements * (1 - self.mask_ratio))

        # Create a binary tensor with the appropriate number of ones
        random_indices = torch.randperm(num_elements)[:num_zeros]
        M = torch.randn(num_elements, device=output.device)
        M[random_indices] = 0
        # Reshape the tensor to match the output shape
        M = M.view(output.shape)

        return self.shaping_module.shape_activation(output, M)

    def register_random_shaping_hooks(self):
        for name, module in self.resnet.named_modules():
            if isinstance(module, nn.Conv2d) and name in self.random_shaping_layers:
                self.hooks_activation_shaping.append(
                    module.register_forward_hook(self.random_shape_activation)
                )

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
