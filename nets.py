import torch
import torch.nn as nn
import torch.nn.functional as F
from activations import GumbelSoftmax, BinaryConcrete



class Encoder(nn.Module):
    def __init__(self, w, k, c, f):
        super().__init__()

        self.final_im_shape = w - (k - 1) * 3                                                   # Image width after 3 convolutions

        self.enc_batch_norm1 = nn.BatchNorm2d(1)                                                # Number of channels
        self.enc_conv1 = nn.Conv2d(1, c, k, bias=False)                                         # In channels, out channels, kernel size
        self.enc_batch_norm2 = nn.BatchNorm2d(c)
        self.enc_conv2 = nn.Conv2d(c, c, k, bias=False)
        self.enc_batch_norm3 = nn.BatchNorm2d(c)
        self.enc_conv3 = nn.Conv2d(c, c, k, bias=False)
        self.enc_batch_norm4 = nn.BatchNorm2d(c)
        self.enc_linear = nn.Linear(c * self.final_im_shape * self.final_im_shape, f)           # Input size, output size

        self.dropout2d = nn.Dropout2d(p=0.2)                                                    # Dropout probability
    
    def forward(self, x):
        x += (0.1**0.5) * torch.randn_like(x)                                                  # Add Gaussian Noise
        #x = self.enc_batch_norm1(x)
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = self.enc_conv3(x)
        '''
        x = self.dropout2d(self.enc_batch_norm2(F.relu(self.enc_conv1(x))))
        x = self.dropout2d(self.enc_batch_norm3(F.relu(self.enc_conv2(x))))
        x = self.dropout2d(self.enc_batch_norm4(self.enc_conv3(x)))
        '''

        x = torch.flatten(x, start_dim=1)                                                       # (batch, final_im_shape)
        x = self.enc_linear(x)                                                                  # (batch, f)
        return x


class Decoder(nn.Module):

    def __init__(self, w, k, c, f, batch):
        super().__init__()

        self.final_im_shape = w - (k - 1) * 3                                                   # Image width after 3 convolutions

        self.dec_linear = nn.Linear(f, c * self.final_im_shape * self.final_im_shape)           # Input size, output size
        self.dec_batch_norm1 = nn.BatchNorm1d(c * self.final_im_shape * self.final_im_shape)
        self.dec_conv1 = nn.ConvTranspose2d(c, c, k, bias=False)                                # In channels, out channels, kernel size
        self.dec_batch_norm2 = nn.BatchNorm2d(c)                                                # Number of channels
        self.dec_conv2 = nn.ConvTranspose2d(c, c, k, bias=False)
        self.dec_batch_norm3 = nn.BatchNorm2d(c)
        self.dec_conv3 = nn.ConvTranspose2d(c, 1, k, bias=False)

        self.dropout2d = nn.Dropout2d(p=0.2)                                                    # Dropout probability

        self.batch = batch                                                                      # For reshaping
        self.c = c

    def forward(self, x):
        x = self.dec_batch_norm1(self.dec_linear(x))                                            # dropout_z false, no 1d dropout
        x = torch.reshape(x, (-1, self.c, self.final_im_shape, self.final_im_shape))
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))

        '''
        x = self.dropout2d(self.dec_batch_norm2(F.relu(self.dec_conv1(x))))
        x = self.dropout2d(self.dec_batch_norm3(F.relu(self.dec_conv2(x))))
        '''

        x = self.dec_conv3(x)
        return x


class Action(nn.Module):

    def __init__(self, f, h1=1000, a=6000):
        super().__init__()
        
        self.linear1 = nn.Linear(f * 2, h1, bias=False)          # Input size, output size
        self.batch_norm1 = nn.BatchNorm1d(h1)
        self.linear2 = nn.Linear(h1, a)

        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        '''
        x = self.dropout(self.batch_norm1(self.relu(self.linear1(x))))
        '''
        x = self.linear2(x)
        return x



class Apply(nn.Module):

    def __init__(self, a, f):
        super().__init__()
        self.linear = nn.Linear(a, f, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(f)
        self.batch_norm2 = nn.BatchNorm1d(f)

    # This is what BtL is - using batch norm transforms the discrete state 'back to logits'
    def forward(self, action, pre_state):
        # action: (batch, a), pre_state: (batch, f)
        action = self.batch_norm1(self.linear(action))                  # action: (batch, f)
        pre_state = self.batch_norm2(pre_state)                         # pre_state: (batch, f)
        apply = action + pre_state
        return apply



class Regress(nn.Module):

    def __init__(self, a, f):
        super().__init__()
        self.linear = nn.Linear(a, f, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(f)
        self.batch_norm2 = nn.BatchNorm1d(f)
    
    def forward(self, action, suc_state):
        # action: (batch, a), suc_state: (batch, f)
        action = self.batch_norm1(self.linear(action))                  # action: (batch, f)
        suc_state = self.batch_norm2(suc_state)                         # suc_state: (batch, f)
        regress = action + suc_state
        return regress
        


class Applicable(nn.Module):
    
    def __init__(self, a, f):
        super().__init__()
        self.linear = nn.Linear(f, a)

    def forward(self, x):
        return self.linear(x)



class Regressable(nn.Module):
    
    def __init__(self, a, f):
        super().__init__()
        self.linear = nn.Linear(f, a)

    def forward(self, x):
        return self.linear(x)