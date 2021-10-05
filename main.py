import torch
import torch.nn as nn

from data.load_data import load_args, load_data
from nets import Autoencoder

"""
Args:
--data          (options: puzzle)

Flags:
-no_cuda        (train on CPU)
-no_wandb       (don't log metrics on wandb)
"""



if __name__=='__main__':

    args = load_args()

    # If usecuda, operations will be performed on GPU
    usecuda = torch.cuda.is_available() and not args.no_cuda
    loaders = load_data(args.data, usecuda=usecuda)

    # Run wandb (logs training)
    if not args.no_wandb:
        import wandb
        run = wandb.init(project='latplan-pytorch',
            group="%s" % (args.data),
            config={'data':args.data},
            reinit=True)
    
    device = torch.device("cuda") if usecuda else torch.device("cpu")
    print("Using device", device)

    # Create model, use Adam optimizer, and use MSE loss
    model = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(30):

        loss = 0
        for data in loaders['train']:

            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            train_loss = criterion(output, data)
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()

        loss = loss / len(loaders['train'])
        print("epoch : {}, loss = {:.6f}".format(epoch + 1, loss))

        # Log training loss in wandb
        if wandb.run is not None:
            wandb.log({"train-loss": train_loss})

    if wandb.run is not None:
        run.finish()