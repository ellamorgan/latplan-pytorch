import torch
import torch.nn as nn
import torch.nn.functional as F


"""
GumbelSoftmax and BinaryConcrete implementations inspired by https://github.com/dev4488/VAE_gumble_softmax/
"""


def get_tau(epoch, t_max=5, t_min=0.5):
    return t_max * (t_min / t_max) ** (min(epoch, 1000) / 1000)



# GumbelSoftmax is for the actions
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



# BinaryConcrete is for the state
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