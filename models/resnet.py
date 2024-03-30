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
      
      M = torch.randint(0, 2, output.shape).cuda()
      total_elements = M.numel()
      num_zeros = int(total_elements * 0.33)
      num_ones = total_elements - num_zeros

      # Reshape the tensor to a 1D tensor
      random_tensor_flat = M.view(-1)

      # Set appropriate number of elements to 0 and 1
      random_tensor_flat[:num_zeros] = 0
      random_tensor_flat[num_zeros:] = 1

      random_tensor_flat = random_tensor_flat[torch.randperm(total_elements)]

      # Reshape the tensor back to its original shape
      M_binary = random_tensor_flat.view(output.shape).float()
 
      # Binarize both A and M using threshold=0 for clarity
      A_binary = (output > 0).float().cuda()

      # Element-wise product for activation shaping
      shaped_output = A_binary * M_binary
      return shaped_output



    def register_hooks(self):
      resnet = self.resnet
      h4 = resnet.layer4[1].conv2.register_forward_hook(self.random_shape_activations)
      return [h4]


    def remove_hooks(self, hooks):
      for h in hooks:
        h.remove()




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
