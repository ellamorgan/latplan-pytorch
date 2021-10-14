import torch
import torch.nn as nn

from data.load_data import load_args, load_data
from model import Model
from loss import total_loss

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

    # This needs to be a tensor on the device
    # Put this in args
    p = torch.Tensor([0.1]).to(device)

    # Create model, use Adam optimizer (the paper uses a different optimizer)
    model = Model(img_width=60, kernel=5, channels=32, fluents=args['f'], batch=args['batch_size'], action_h1=1000, action=6000, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Training loop
    for epoch in range(args['epochs']):

        train_loss = 0

        for data in loaders['train']:

            data = data.to(device)
            optimizer.zero_grad()
            out = model(data, epoch)
            loss = total_loss(out, p)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= (len(loaders['train']) * args['batch_size'])
        print("epoch: {}, train loss = {:.6f}, ".format(epoch + 1, train_loss), end="")

        # Validation
        with torch.no_grad():

            model.eval()
            val_loss = 0

            for data in loaders['val']:

                data = data.to(device)
                out = model(data, epoch)
                loss = total_loss(out, p)
                val_loss += loss.item()
            
            val_loss /= (len(loaders['val']) * args['batch_size'])
            print("val loss = {:.6f}".format(val_loss))
        
            model.train()

        # Log results in wandb
        if wandb is not None:
            wandb.log({"train-loss": train_loss, "val-loss": val_loss})

    if wandb is not None:
        run.finish()