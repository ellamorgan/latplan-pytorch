import torch
import torch.nn as nn
import torch.nn.functional as F



class Encoder(nn.Module):

    def __init__(self, w=60, k=5, c = 32, f=100):
        """
        Takes in image width/height (w), kernel size (k), number of channels (c), and latent space dimension (f)
        """
        super().__init__()

        self.batch_norm1 = nn.BatchNorm2d(1)            # Number of channels
        self.conv1 = nn.Conv2d(1, c, k, bias=False)     # In channels, out channels, kernel size
        self.batch_norm2 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, k, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(c)
        self.conv3 = nn.Conv2d(c, c, k, bias=False)
        self.batch_norm4 = nn.BatchNorm2d(c)
        self.dropout = nn.Dropout2d(p=0.2)              # Dropout probability

        # Input size is width - (kernel_size - 1) * 3
        # where 3 is the number of convolution layers
        self.linear = nn.Linear(c * (w - (k - 1) * 3) * (w - (k - 1) * 3), f)     # Input size, output size

    def forward(self, x):
        # x: (batch, channels, wight, height)
        x = self.batch_norm1(x)
        x = self.dropout(self.batch_norm2(F.relu(self.conv1(x))))
        x = self.dropout(self.batch_norm3(F.relu(self.conv2(x))))
        x = self.dropout(self.batch_norm4(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)                           # (batch, 32 * (w - (k - 1) * 3))
        x = self.linear(x)                                          # (batch, f)

        # Apply binary concrete activation after to make discrete
        #z = binary_concrete(l)

        return x



class Decoder(nn.Module):

    def __init__(self, w=60, k=5, c = 32, f=100):
        super().__init__()

        self.linear = nn.Linear(f, c * (w - (k - 1) * 3) * (w - (k - 1) * 3))
        self.batch_norm1 = nn.BatchNorm1d(c * (w - (k - 1) * 3) * (w - (k - 1) * 3))
        self.conv1 = nn.ConvTranspose2d(c, c, k, bias=False)   # In channels, out channels, kernel size
        self.batch_norm2 = nn.BatchNorm2d(c)                   # Number of channels
        self.conv2 = nn.ConvTranspose2d(c, c, k, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(c)
        self.conv3 = nn.ConvTranspose2d(c, 1, k, bias=False)
        self.dropout1d = nn.Dropout(p=0.2)
        self.dropout2d = nn.Dropout2d(p=0.2)                      # Dropout probability
    
    def forward(self, x):
        x = self.dropout1d(self.batch_norm1(self.linear(x)))
        x = torch.reshape(x, (5, 32, 48, 48))
        x = self.dropout2d(self.batch_norm2(F.relu(self.conv1(x))))
        x = self.dropout2d(self.batch_norm3(F.relu(self.conv2(x))))
        x = self.conv3(x)
        return x



# Combines Encoder and Decoder networks
class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

'''
class Action(nn.Module):

    def __init__(self):
        super().__init__()
        # Define linear layer
    
    def forward(self, x):
        # Gumbel softmax - define this activation
        # Figure out how to define an activation function in pytorch
        # Some weird reshaping occurs between the layer and the activation
        # (They call this reshaping "adim" in the DetActionMixin class)
        #x = gumbel_softmax(linear_layer(x))
'''

# TODO
# Loss functions, different KL-D losses for Binary Concrete and Gumbel Softmax
# In the code these are found in the AbstractBinaryConcrete and AbstractGumbelSoftmax classes
# The full KL-D loss also includes the reconstruction loss, in the code they use GaussianOutput from output.py
# I think Gaussian Output is just negative log likelihood?


# Gumbel Softmax
# Applied to the action (formula, not network)

# Binary Concrete
# Turns logits into discrete (formula, not network)
# For GS and BC they do some weird sampling stuff

# Action
# Predicts action based on l^{i, 0} and l^{i, 1}
# l is the output of the encoder before applying BC

# Apply
# This is a formula, logically computes the next state

# Regress
# Predicts z^0 from z^1 and a

# Applicable?
# Regressable?
# In loss