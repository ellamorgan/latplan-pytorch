import torch
import torch.nn as nn
import torch.nn.functional as F

from nets import Encoder, Decoder, Action, Apply, Regress, Applicable, Regressable
from activations import GumbelSoftmax, BinaryConcrete

"""
Full model, follows flow provided in section 3.1 Model descriptions
"""


class Model(nn.Module):

    def __init__(self, img_width, kernel, channels, fluents, batch, action_h1, action, device, **kwargs):
        super().__init__()

        self.w = img_width
        self.k = kernel
        self.c = channels
        self.f = fluents
        self.b = batch
        self.h1 = action_h1
        self.a = action

        self.device = device

        self.encoder = Encoder(self.w, self.k, self.c, self.f)
        self.decoder = Decoder(self.w, self.k, self.c, self.f, self.b)
        self.action = Action(self.f, self.h1, self.a)
        self.apply_action = Apply(self.a, self.f)
        self.regress = Regress(self.a, self.f)
        self.applicable = Applicable(self.a, self.f)
        self.regressable = Regressable(self.a, self.f)

        self.gumbel_softmax = GumbelSoftmax(self.device)
        self.binary_concrete = BinaryConcrete(self.device)
    
    
    def forward(self, x, epoch):

        # x is a pair of images. x[:, 0] is pre, x[:, 1] is suc
        out = {'x_0': x[:, 0], 'x_1': x[:, 1]}

        out['l_0'] = self.encoder(out['x_0'])
        out['l_1'] = self.encoder(out['x_1'])

        out['z_0'] = self.binary_concrete(out['l_0'], epoch)
        out['z_1'] = self.binary_concrete(out['l_1'], epoch)

        out['a_l'] = self.action(torch.cat((out['l_0'], out['l_1']), axis=1))
        out['a'] = self.gumbel_softmax(out['a_l'], epoch)

        out['l_2'] = self.apply_action(out['a'], out['z_0'])
        out['l_3'] = self.regress(out['a'], out['z_1'])

        out['z_2'] = self.binary_concrete(out['l_2'], epoch)
        out['z_3'] = self.binary_concrete(out['l_3'], epoch)

        out['x_dec_0'] = self.decoder(out['z_0'])
        out['x_dec_1'] = self.decoder(out['z_1'])
        out['x_aae_2'] = self.decoder(out['z_2'])
        out['x_aae_3'] = self.decoder(out['z_3'])

        out['app'] = self.applicable(out['z_0'])
        out['reg'] = self.regressable(out['z_1'])

        return out