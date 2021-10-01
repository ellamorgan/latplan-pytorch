import torch
import torch.nn as nn
import torch.nn.functional as F

# Not sure if these all belong in one file, some might make sense to combine under one class. Not sure

# Encoder
# Takes in x^{i, 0} and x^{i, 1}, returns l^{i, 0} and l^{i, 1}
class Encoder(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)                 # In channels, out channels, kernel size
        self.pool = nn.MaxPool2d(2, 2)                  # Kernel size, stride
        self.conv2 = nn.Conv2d(6, 16, 5)                # In channels, out channels, kernel size
        self.fc1 = nn.Linear(16 * 5 * 5, 120)           # In features, out features
        self.fc2 = nn.Linear(120, 84)                   # In features, out features
        self.fc3 = nn.Linear(84, 10)                    # In features, out features
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Decoder
# Decodes discrete state back to image

# Gumbel Softmax
# Applied to the action (formula, not network)

# Binary Concrete
# Turns logits into discrete (formula, not network)

# In the original implementation there's an Autoencoder class with encode, decode, autoencode, autodecode, build_gs, and build_gc functions
# This class inherits the class 'Network' which handles the forward passes and such
# Network has a list of nets, optimizers, etc with elements corresponding to the subnetworks
# Then each subnetwork (ex. encode, decode) also has its own class
# We can have high level classes like this, then individual classes for Encode, Decode, BinaryConcrete, GumbelSoftmax?
class Autoencoder:

    def __init__():
        pass

    def encode(self):
        return None
    
    def decode(self):
        return None
    
    def binary_concrete(self):
        return None
    
    def gumbel_softmax(self):
        return None


# Action
# Predicts action based on l^{i, 0} and l^{i, 1}

# Apply
# This is a formula, logically computes the next state

# Regress
# Predicts z^0 from z^1 and a

# Applicable?
# Regressable?
# What are these? They're implemented as FC layers, the paper states they regularize the action
# and they're only used during training?
# How do they regularize the action? The paper barely mentions them
# There's no instances of 'applicable' or 'regressable' in the original code, very weird
# If they're just for regularization we should be able to get the model working without them

class Action:
    
    def __init__():
        pass

    def action():
        return None
    
    def apply():
        return None
    
    def regress():
        return None
    
    def applicable():
        return None
    
    def regressable():
        return None