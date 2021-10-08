import torch
import torch.nn as nn
import torch.nn.functional as F



# Combines Encoder and Decoder networks
# Do the linear layers have bias?
# Need to check exsct specifications (ex. initialization is wrong)
class Autoencoder(nn.Module):

    def __init__(self, w, k, c, f, batch):
        super().__init__()

        self.final_im_shape = w - (k - 1) * 3                                                   # Image width after 3 convolutions

        # Encoder
        self.enc_batch_norm1 = nn.BatchNorm2d(1)                                                # Number of channels
        self.enc_conv1 = nn.Conv2d(1, c, k, bias=False)                                         # In channels, out channels, kernel size
        self.enc_batch_norm2 = nn.BatchNorm2d(c)
        self.enc_conv2 = nn.Conv2d(c, c, k, bias=False)
        self.enc_batch_norm3 = nn.BatchNorm2d(c)
        self.enc_conv3 = nn.Conv2d(c, c, k, bias=False)
        self.enc_batch_norm4 = nn.BatchNorm2d(c)
        self.enc_linear = nn.Linear(c * self.final_im_shape * self.final_im_shape, f)           # Input size, output size

        # Decoder
        self.dec_linear = nn.Linear(f, c * self.final_im_shape * self.final_im_shape)           # Input size, output size
        self.dec_batch_norm1 = nn.BatchNorm1d(c * self.final_im_shape * self.final_im_shape)
        self.dec_conv1 = nn.ConvTranspose2d(c, c, k, bias=False)                                # In channels, out channels, kernel size
        self.dec_batch_norm2 = nn.BatchNorm2d(c)                                                # Number of channels
        self.dec_conv2 = nn.ConvTranspose2d(c, c, k, bias=False)
        self.dec_batch_norm3 = nn.BatchNorm2d(c)
        self.dec_conv3 = nn.ConvTranspose2d(c, 1, k, bias=False)

        self.dropout1d = nn.Dropout(p=0.2)                                                      # Dropout probability
        self.dropout2d = nn.Dropout2d(p=0.2)

        self.batch = batch
        self.c = c
    
    def encode(self, x):
        x = self.enc_batch_norm1(x)
        x = self.dropout2d(self.enc_batch_norm2(F.relu(self.enc_conv1(x))))
        x = self.dropout2d(self.enc_batch_norm3(F.relu(self.enc_conv2(x))))
        x = self.dropout2d(self.enc_batch_norm4(self.enc_conv3(x)))
        x = torch.flatten(x, start_dim=1)                                                       # (batch, final_im_shape)
        x = self.enc_linear(x)                                                                  # (batch, f)
        return x
    
    def decode(self, x):
        x = self.dec_batch_norm1(self.dec_linear(x))                                            # dropout_z false, no 1d dropout
        x = torch.reshape(x, (self.batch, self.c, self.final_im_shape, self.final_im_shape))
        x = self.dropout2d(self.dec_batch_norm2(F.relu(self.dec_conv1(x))))
        x = self.dropout2d(self.dec_batch_norm3(F.relu(self.dec_conv2(x))))
        x = self.dec_conv3(x)
        return x

    def forward(self, x):
        # x: (batch, channels, wight, height)
        l = self.encode(x)
        # l: (batch, f)
        x = self.decode(l)
        # x: (batch, channels, wight, height)
        return l, x



class Action(nn.Module):

    def __init__(self, f, h1=1000, a=6000):
        super().__init__()
        
        self.linear1 = nn.Linear(f * 2, h1)          # Input size, output size
        self.batch_norm1 = nn.BatchNorm1d(h1)
        self.linear2 = nn.Linear(h1, a)

        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
    
    def gumbel_softmax(self, x):
        return x

    def forward(self, x):
        x = self.dropout(self.batch_norm1(self.relu(self.linear1(x, bias=False))))
        x = self.gumbel_softmax(self.linear2(x))
        return x



# TODO
# Loss functions, different KL-D losses for Binary Concrete and Gumbel Softmax
# In the code these are found in the AbstractBinaryConcrete and AbstractGumbelSoftmax classes
# The full KL-D loss also includes the reconstruction loss, in the code they use GaussianOutput from output.py


# Gumbel Softmax
# Applied to the action (formula, not network)

# Binary Concrete
# Turns logits into discrete (formula, not network)
# For GS and BC they do some weird sampling stuff

# Apply
# This is a formula, logically computes the next state

# Regress
# Predicts z^0 from z^1 and a

# Applicable?
# Regressable?
# In loss