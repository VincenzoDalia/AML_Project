import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class BaseResNet18(nn.Module):
    def __init__(self):
        super(BaseResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

    def forward(self, x):
        return self.resnet(x)


# Modifies 'BaseResNet18' including the Activation Shaping Module
class ASHResNet18(nn.Module):
    def __init__(self):
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

    def forward(self, x):
        return self.resnet(x)

    def random_shape_activations(self, model, input, output):

        output_shape = output.shape
        num_elements = output.numel()
        num_ones = int(num_elements * 0.5)
        # Create a binary tensor with the appropriate number of ones
        random_indices = torch.randperm(num_elements)[:num_ones]
        M_binary = torch.zeros(num_elements, device=output.device)
        M_binary[random_indices] = 1
        # Reshape the tensor to match the output shape
        M_binary = M_binary.view(output_shape)

        A_binary = (output > 0).float()

        # Element-wise product for activation shaping
        shaped_activation = A_binary * M_binary

        return shaped_activation

    def register_hooks(self):
        resnet = self.resnet
        h3 = resnet.layer3[1].conv2.register_forward_hook(self.random_shape_activations)
        h4 = resnet.layer3[0].conv2.register_forward_hook(self.random_shape_activations)
        return [h3, h4]

    def remove_hooks(self, hooks):
        for h in hooks:
            h.remove()


######################################################
# TODO: either define the Activation Shaping Module as a nn.Module
# class ActivationShapingModule(nn.Module):
# ...
#
# OR as a function that shall be hooked via 'register_forward_hook'
# def activation_shaping_hook(module, input, output):
# ...
#
######################################################
# TODO: modify 'BaseResNet18' including the Activation Shaping Module
# class ASHResNet18(nn.Module):
#    def __init__(self):
#        super(ASHResNet18, self).__init__()
#        ...
#
#    def forward(self, x):
#        ...
#
######################################################
