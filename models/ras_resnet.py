import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# Modifies 'BaseResNet18' including the Activation Shaping Module
class RASResNet18(nn.Module):
    def __init__(
        self,
        mask_ratio,
        use_bernoulli,
        shaping_module,
        random_shape_layers=["layer2.1.conv2"],
    ):
        super(RASResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.shaping_module = shaping_module
        self.random_shape_layers = random_shape_layers
        self.hooks = []
        self.mask_ratio = mask_ratio
        self.use_bernoulli = use_bernoulli

    def forward(self, x):
        return self.resnet(x)

    def random_shape_activation(self, model, input, output):
        if self.use_bernoulli:
            return self.shape_with_bernoulli(output)

        return self.shape_with_exact_ratio(output)

    def shape_with_bernoulli(self, output):
        """
        Use this faster function if no_binarize is False
        """
        p = torch.full_like(output, self.mask_ratio)
        M = torch.bernoulli(p)

        return self.shaping_module.shape_activation(output, M)

    def shape_with_exact_ratio(self, output):
        num_elements = output.numel()
        num_zeros = int(num_elements * (1 - self.mask_ratio))

        M = torch.rand(num_elements, device=output.device)

        # Create a binary tensor with the appropriate number of ones
        random_indices = torch.randperm(num_elements)[:num_zeros]
        M[random_indices] = 0

        # Reshape the tensor to match the output shape
        M = M.view(output.shape)

        return self.shaping_module.shape_activation(output, M)

    def register_random_shaping_hooks(self):
        for name, module in self.resnet.named_modules():
            if name in self.random_shape_layers:
                self.hooks.append(
                    module.register_forward_hook(self.random_shape_activation)
                )

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
