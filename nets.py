import torch
import torch.nn as nn
import torch.nn.functional as F



# Adding KL divergence requires changing this to a 'VariationalEncoder' I believe
class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.batch_norm1 = nn.BatchNorm2d(1)            # Number of channels
        self.conv1 = nn.Conv2d(1, 32, 5, bias=False)    # In channels, out channels, kernel size
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 5, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 5, bias=False)
        self.dropout = nn.Dropout2d(p=0.2)              # Dropout probability
    
    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout(self.batch_norm2(F.relu(self.conv1(x))))
        x = self.dropout(self.batch_norm3(F.relu(self.conv2(x))))
        x = self.conv3(x)
        return x



class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.batch_norm1 = nn.BatchNorm2d(32)                   # Number of channels
        self.conv1 = nn.ConvTranspose2d(32, 32, 5, bias=False)  # In channels, out channels, kernel size
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.conv2 = nn.ConvTranspose2d(32, 32, 5, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.conv3 = nn.ConvTranspose2d(32, 1, 5, bias=False)
        self.dropout = nn.Dropout2d(p=0.2)                      # Dropout probability
    
    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout(self.batch_norm2(F.relu(self.conv1(x))))
        x = self.dropout(self.batch_norm3(F.relu(self.conv2(x))))
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


# TODO
# Gumbel Softmax
# Applied to the action (formula, not network)

# Binary Concrete
# Turns logits into discrete (formula, not network)

# Action
# Predicts action based on l^{i, 0} and l^{i, 1}

# Apply
# This is a formula, logically computes the next state

# Regress
# Predicts z^0 from z^1 and a

# Applicable?
# Regressable?
# These are FC layers that "regularize the action" but only during training?
# No other info is given and I can't find their existence in the original code