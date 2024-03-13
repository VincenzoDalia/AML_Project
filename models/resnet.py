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
    
    def register_activation_shaping_hook(self, module, every_nth=1):
        """
        Registers the activation shaping hook to a specific module.

        Args:
            module (torch.nn.Module): The module to which the hook will be attached.
            every_nth (int, optional): Apply the hook every nth convolution.
                Defaults to 3.
        """

        def activation_shaping_hook(module, input, output):
            
            
            print("Activation Shaping Hook")
            
        # Handle applying the hook every nth convolution if specified
        if isinstance(module, nn.Conv2d):
            if every_nth == 1:
                handle = module.register_forward_hook(activation_shaping_hook)
                self.registered_hooks.append(handle)  # Optional for clarity
            else:
                count = 0
                def wrapper(module, input, output):
                    nonlocal count
                    count += 1
                    if count % every_nth == 0:
                        activation_shaping_hook(module, input, output)
                    return output
                handle = module.register_forward_hook(wrapper)
                self.registered_hooks.append(handle)  # Optional for clarity

######################################################
# TODO: either define the Activation Shaping Module as a nn.Module
#class ActivationShapingModule(nn.Module):
#...
#
# OR as a function that shall be hooked via 'register_forward_hook'
#def activation_shaping_hook(module, input, output):
#...
#
######################################################
# TODO: modify 'BaseResNet18' including the Activation Shaping Module
#class ASHResNet18(nn.Module):
#    def __init__(self):
#        super(ASHResNet18, self).__init__()
#        ...
#    
#    def forward(self, x):
#        ...
#
######################################################
