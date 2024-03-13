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
        #self.registered_hooks = []

    def forward(self, x):
        return self.resnet(x)
