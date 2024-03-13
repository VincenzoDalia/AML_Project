""" import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class BaseResNet18(nn.Module):
    def init(self):
        super(BaseResNet18, self).init()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

    def forward(self, x):
        return self.resnet(x)

    ###################   PUNTO 1  #######################
    # TODO: either define the Activation Shaping Module as a nn.Module
    #class ActivationShapingModule(nn.Module):
    #...
    #
    # OR as a function that shall be hooked via 'register_forward_hook'
    def activation_shaping_hook(module, input, output):
    

        # ricevo A ed M come input, ne faccio la binarizzazione
        pass
    
    
    
######################################################
# TODO: modify 'BaseResNet18' including the Activation Shaping Module
#class ASHResNet18(nn.Module):
#    def init(self):
#        super(ASHResNet18, self).init()
#        ...
#    
#    def forward(self, x):
#        ...
#
###################################################### """


import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class BaseResNet18(nn.Module):
    def init(self):
        super(BaseResNet18, self).init()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

        # Create an empty list to store registered hooks (optional for clarity)
        self.registered_hooks = []

    def forward(self, x):
        return self.resnet(x)

    def register_activation_shaping_hook(self, module, every_nth=3):
        """
        Registers the activation shaping hook to a specific module.

        Args:
            module (torch.nn.Module): The module to which the hook will be attached.
            every_nth (int, optional): Apply the hook every nth convolution.
                Defaults to 3.
        """

        def activation_shaping_hook(module, input, output):
         
            M = torch.randn(output.shape)
            
            # Binarize both A and M using threshold=0 for clarity
            A_binary = (output > 0).float()
            M_binary = (M > 0).float()

            # Element-wise product for activation shaping
            shaped_output = A_binary * M_binary

            # Replace the original output with the shaped output
            module.out_func = lambda input, output: shaped_output

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

    def remove_activation_shaping_hooks(self):
        """
        Removes all registered activation shaping hooks.
        """
        for handle in self.registered_hooks:
            handle.remove()
        self.registered_hooks = []