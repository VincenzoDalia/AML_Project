import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class BaseResNet18(nn.Module):
    def __init__(self):
        super(BaseResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.counter = 0

    def forward(self, x):
        return self.resnet(x)


# Modifies 'BaseResNet18' including the Activation Shaping Module
class ASHResNet18(nn.Module):
    def __init__(self):
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.hooks = []
        
        self.binary = False
        self.topK = False

    def forward(self, x):
        return self.resnet(x)

    # Generic module that performs activation shaping
    def shape_activation(self, layer_activation, M):
        
        
        if self.topK:
            # Select the top K values of A (output)
            
            M_binary = (M > 0).float()
            
            K = 5
            
            #EXTENSION 2.b
            
            top_values, top_indices = torch.topk(layer_activation.flatten(), k=3)

            # Creare un nuovo tensore con tutti i valori a 0
            A_topK = torch.zeros_like(layer_activation)

            # Assegna i valori originali ai loro indici corrispondenti
            A_topK.view(-1)[top_indices] = layer_activation.view(-1)[top_indices]

            return A_topK * M_binary
        
        elif self.binary:
            M_binary = (M > 0).float()
            A_binary = (layer_activation > 0).float()
            
            # Element-wise product for activation shaping
            return A_binary * M_binary
        
        else:
            # Extension 2.a (Binarization Ablation for M)
            #A_binary = (layer_activation > 0).float()
            
            return layer_activation * M
    
        

        

    def random_shape_activations(self, mask_ratio):
        def hook(model, input, output):
            
             
            num_elements = output.numel()
            num_zeros = int(num_elements * (1-mask_ratio))
            # Create a binary tensor with the appropriate number of ones
            random_indices = torch.randperm(num_elements)[:num_zeros]
            M = torch.randn(num_elements, device=output.device)
            M[random_indices] = 0

            # Reshape the tensor to match the output shape
            M = M.view(output.shape)

            return self.shape_activation(output, M)
            

        return hook

    # TODO: domain_adapt_activations():

    def register_random_shaping_hooks(self, mask_ratio):
        random_maps_hook = self.random_shape_activations(mask_ratio)

        h2 = self.resnet.layer2[1].conv2.register_forward_hook(random_maps_hook)
        # h3 = self.resnet.layer3[0].conv2.register_forward_hook(random_maps_hook)

        # self.hooks.append(h3)
        self.hooks.append(h2)

    # TODO: register_domain_adapt_hooks()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


# Modifies 'BaseResNet18' including the Domain Adaptation Module
# TODO: Implement Domain Adaptation Module
class DomAdaptResNet18(nn.Module):
    def __init__(self):
        super(DomAdaptResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        
        self.hooks_activation_maps = []
        self.hooks_activation_shaping = []
        self.activation_maps = []
        
        self.binary = True
        self.topK = False
        #List of layers to be activated
        self.active_layers = ['layer2.1.conv2']
        
    
    #To save the activation maps
    def register_map_hooks(self):
        for name, module in self.resnet.named_modules():
            if isinstance(module, nn.Conv2d) and name in self.active_layers:
                self.hooks_activation_maps.append(module.register_forward_hook(self.save_maps))
                
    #To append the activation maps to the list       
    def save_maps(self, module, input, output):
        self.activation_maps.append(output.clone().detach())
        
    #To do the activation shaping
    def activation_shaping(self, model, input, output):
        
        M = self.activation_maps.pop(0)
        
        
        if self.topK:
            # Select the top K values of A (output)
            
            M_binary = (M > 0).float()
            
            K = 5
            
            #EXTENSION 2.b
            
            top_values, top_indices = torch.topk(output.flatten(), k=3)

            # Creare un nuovo tensore con tutti i valori a 0
            A_topK = torch.zeros_like(output)

            # Assegna i valori originali ai loro indici corrispondenti
            A_topK.view(-1)[top_indices] = output.view(-1)[top_indices]

            """
            # Stampa i tensori originali e modificati
            print("Tensore originale:")
            print(output)
            print("\nTensore modificato:")
            print(A_topK)
             """
            
            return A_topK * M_binary
        
        elif self.binary:
            M_binary = (M > 0).float()
            A_binary = (output > 0).float()
            
            # Element-wise product for activation shaping
            return A_binary * M_binary
        
        else:
            # Extension 2.a (Binarization Ablation for M)
            #A_binary = (output > 0).float()
            
            return output * M
      
    
    #To register the activation shaping hooks
    def register_shaping_hooks(self):
       
        for name, module in self.resnet.named_modules():
            if isinstance(module, nn.Conv2d) and name in self.active_layers:
                self.hooks_activation_shaping.append(module.register_forward_hook(self.activation_shaping))
        
        
    def forward(self, x):
        return self.resnet(x)
    
    
    def remove_hooks_activation_maps(self):
        for h in self.hooks_activation_maps:
            h.remove()
     
    def remove_hooks_activation_shaping(self):
        for h in self.hooks_activation_shaping:
            h.remove()

