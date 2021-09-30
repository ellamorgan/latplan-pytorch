# Not sure if these all belong in one file, some might make sense to combine under one class. Not sure

# In the original implementation there's an Autoencoder class with encode, decode, autoencode, autodecode, build_gs, and build_gc functions
# This class inherits the class 'Network' which handles the forward passes and such
# Network has a list of nets, optimizers, etc with elements corresponding to the subnetworks
# Then each subnetwork (ex. encode, decode) also has its own class
class Autoencoder:

    def __init__():
        pass

    def encode(self):
        return None
    
    def decode(self):
        return None
    
    def build_bc(self):
        return None
    
    def build_gs(self):
        return None


# Encoder
# Takes in x^{i, 0} and x^{i, 1}, returns l^{i, 0} and l^{i, 1}

# Action
# Predicts action based on l^{i, 0} and l^{i, 1}

# Gumbel Softmax
# Applied to the action (formula, not network)

# Binary Concrete
# Turns logits into discrete (formula, not network)

# Decoder

# Apply
# This is a formula, logically computes the next state

# Regress
# Predicts z^0 from z^1 and a

# Decode
# Decodes discrete state back to image

# Applicable?
# Regressable?
# What are these? They're implemented as FC layers, the paper states they regularize the action
# and they're only used during training?
# How do they regularize the action? The paper barely mentions them
# There's no instances of 'applicable' or 'regressable' in the original code, very weird
# If they're just for regularization we should be able to get the model working without them