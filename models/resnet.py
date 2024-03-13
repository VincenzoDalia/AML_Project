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
    


    def activation_shaping_hook(self, module, input, output):
        
            M = torch.randn(output.shape).cuda()
            M = torch.Tensor(M.data)
            M.requires_grad = True
            
            # Binarize both A and M using threshold=0 for clarity
            A_binary = (output > 0).float().cuda()
            M_binary = (M > 0).float()

            # Element-wise product for activation shaping
            shaped_output = A_binary * M_binary
            
            return shaped_output

            

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
