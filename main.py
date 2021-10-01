import torch
#import wandb

from data.load_data import *
from nets import *

"""
Args:
--data          (puzzle-3, puzzle-4, puzzle-5, will add more)

Flags:
-no_cuda        (train on CPU)
-no_wandb       (don't log metrics on wandb)
"""

if __name__=='__main__':

    args = load_args()

    # If usecuda, operations will be performed on GPU
    #usecuda = torch.cuda.is_available() and not args.no_cuda
    #train_loader, val_loader, test_loader = load_data(args.data, usecuda)
    load_data(args.data)