import torch
import torch.nn as nn
import torch.nn.functional as F

from nets import Autoencoder, Action

"""
Full model
"""


class Model(nn.Module):

    def __init__(self, img_width, kernel, channels, fluents, batch, action_h1, action):
        super().__init__()

        self.w = img_width
        self.k = kernel
        self.c = channels
        self.f = fluents
        self.b = batch
        self.h1 = action_h1
        self.a = action

        self.AE = Autoencoder(self.w, self.k, self.c, self.f, self.batch)
        self.Action = Action(self.f, self.h1, self.a)
    
    def forward(self, x):
