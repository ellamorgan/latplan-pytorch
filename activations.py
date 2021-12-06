import torch
import torch.nn as nn
import torch.nn.functional as F


"""
GumbelSoftmax and BinaryConcrete implementations inspired by https://github.com/dev4488/VAE_gumble_softmax/
"""


# Calculates tau - formula provided in section 3.1.6 Gumbel Softmax
def get_tau(epoch, t_max=5, t_min=0.5, total_epochs=1000):
    return t_max * (t_min / t_max) ** (min(epoch, total_epochs) / total_epochs)



# GumbelSoftmax is for the actions - discretizes as tau approaches 0
class GumbelSoftmax(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
    
    def forward(self, x, epoch, eps=1e-20):
        # x: (batch, a)
        tau = get_tau(epoch)
        u = torch.rand(x.shape, device=self.device)
        gumbel = -torch.log(-torch.log(u + eps) + eps)
        logits = (x + gumbel) / tau
        gumbel_softmax = F.softmax(logits, dim=-1)
        return x



# BinaryConcrete is for the state - discretizes as tau approaches 0
class BinaryConcrete(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
    
    def forward(self, x, epoch, eps=1e-20):
        # x: (batch, f)
        tau = get_tau(epoch)
        u = torch.rand(x.shape, device=self.device)
        logistic = torch.log(u + eps) - torch.log(1 - u + eps)
        logits = (x + logistic) / tau
        binary_concrete = torch.sigmoid(logits)
        return binary_concrete