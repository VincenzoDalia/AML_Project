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
        self.hooks = []

    def forward(self, x):
        return self.resnet(x)

    # Generic module that performs activation shaping
    def shape_activation(self, layer_activation, M):
        # Binarize both A and M
        A_binary = (layer_activation > 0).float()
        
        # consider removing M binarization here
        # random_shaping_hook can binarize instead for point 2
        # ext 2 requires retaining M as it is
        M_binary = (M > 0).float()

        # Element-wise product for activation shaping
        return A_binary * M


    def random_shape_activations(self, mask_ratio):
        def hook(model, input, output):
          num_elements = output.numel()
          num_ones = int(num_elements * mask_ratio)
          # Create a binary tensor with the appropriate number of ones
          random_indices = torch.randperm(num_elements)[:num_ones]
          M_binary = torch.zeros(num_elements, device=output.device)
          M_binary[random_indices] = 1
          # Reshape the tensor to match the output shape
          M_binary = M_binary.view(output.shape)
          
          return self.shape_activation(output, M_binary)
        
        return hook

    def register_random_shaping_hooks(self, mask_ratio):
        random_maps_hook = self.random_shape_activations(mask_ratio)
        
        h2 = self.resnet.layer2[1].conv2.register_forward_hook(random_maps_hook)
        #h3 = self.resnet.layer3[0].conv2.register_forward_hook(random_maps_hook)
        
        #self.hooks.append(h3)
        self.hooks.append(h2)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()