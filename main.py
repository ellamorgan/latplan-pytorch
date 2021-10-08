import torch
import torch.nn as nn

from data.load_data import load_args, load_data
from nets import Autoencoder

"""
Args:
--dataset       (options: puzzle)

Flags:
-no_cuda        (train on CPU)
-no_wandb       (don't log metrics on wandb)
"""



if __name__=='__main__':

    args = load_args('configs.conf')

    # If usecuda, operations will be performed on GPU
    usecuda = torch.cuda.is_available() and not args['no_cuda']
    loaders = load_data(**args, usecuda=usecuda)

    # Run wandb (logs training)
    if not args['no_wandb']:
        import wandb
        run = wandb.init(project='latplan-pytorch',
            group="%s" % (args['dataset']),
            config={'dataset':args['dataset']},
            reinit=True)
    else:
        wandb = None
    
    device = torch.device("cuda") if usecuda else torch.device("cpu")
    print("Using device", device)

    # Create model, use Adam optimizer, and use MSE loss
    model = Autoencoder(w=60, k=5, c = 32, f=args['f'], batch=args['batch_size']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(args['epochs']):

        train_loss = 0

        for data in loaders['train']:

            data = data.to(device)
            optimizer.zero_grad()
            pre_logits, pre_output = model(data[:, 0])                  # logits: (batch, f)
            suc_logits, suc_output = model(data[:, 1])                  # output: (batch, 1, w, w)

            action_inp = torch.cat((pre_logits, suc_logits), axis=1)    # action_inp: (batch, 2 * f)

            # Loss functions go here
            loss = criterion(pre_output, data[:, 0]) + criterion(suc_output, data[:, 1])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(loaders['train'])
        print("epoch: {}, train loss = {:.6f}, ".format(epoch + 1, train_loss), end="")

        # Validation
        with torch.no_grad():

            model.eval()
            val_loss = 0

            for data in loaders['val']:

                data = data.to(device)
                pre_logits, pre_output = model(data[:, 0])
                suc_logits, suc_output = model(data[:, 1])
                loss = criterion(pre_output, data[:, 0]) + criterion(suc_output, data[:, 1])
                val_loss += loss.item()
            
            val_loss /= len(loaders['val'])
            print("val loss = {:.6f}".format(val_loss))
        
            model.train()

        # Log results in wandb
        if wandb is not None:
            wandb.log({"train-loss": train_loss, "val-loss": val_loss})

    if wandb is not None:
        run.finish()