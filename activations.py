import torch
import torch.nn as nn
import torch.nn.functional as F


"""
GumbelSoftmax and BinaryConcrete implementations inspired by https://github.com/dev4488/VAE_gumble_softmax/
"""



# GumbelSoftmax is for the actions
class GumbelSoftmax(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, x, tau, device, eps=1e-20):
        # x: (batch, a)
        u = torch.rand(x.shape, device=device)
        gumbel = -torch.log(-torch.log(u + eps) + eps)
        logits = (x + gumbel) / tau
        gumbel_softmax = F.softmax(logits, dim=-1)
        return x



# BinaryConcrete is for the state
class BinaryConcrete(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, x, tau, device, eps=1e-20):
        # x: (batch, f)
        u = torch.rand(x.shape, device=device)
        logistic = torch.log(u + eps) - torch.log(1 - u + eps)
        logits = (x + logistic) / tau
        binary_concrete = F.sigmoid(logits, dim=-1)
        return binary_concrete