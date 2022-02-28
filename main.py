import torch
import torch.nn as nn
import numpy as np
import itertools

from data.load_data import load_args, load_data
from model import Model
from loss import total_loss
from output import save_latents_as_gif, save_loss_plots
from interpretability import neuron_attribution, save_latent, generate_heatmap

"""
Flags:
-no_cuda        (train on CPU)
-no_wandb       (don't log metrics on wandb)
"""


def train(args, model, device, loaders, losses):

    # Run wandb (logs training)
    if not args['no_wandb']:
        import wandb
        run = wandb.init(project='latplan-pytorch',
            group="%s" % (args['dataset']),
            config={'dataset':args['dataset']},
            reinit=True)
    else:
        wandb = None

    # Need to put this tensor on the device, done here instead of passing device
    p = torch.Tensor([args['p']]).to(device)
    
    optimizer = torch.optim.RAdam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    # Training loop
    for epoch in range(args['epochs']):

        train_loss = 0

        for data in loaders['train']:

            data = data.to(device)
            optimizer.zero_grad()
            out = model(data, epoch)
            loss, losses = total_loss(out, p, args['beta_z'], args['beta_d'], losses)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= (len(loaders['train']) * args['batch'])
        print("epoch: {}, train loss = {:.6f}, ".format(epoch + 1, train_loss), end="")

        # Validation - not used to impact model training, essentially working as a testing dataset
        with torch.no_grad():

            model.eval()
            val_loss = 0

            for data in loaders['val']:

                data = data.to(device)
                out = model(data, epoch)
                loss, _ = total_loss(out, p, args['beta_z'], args['beta_d'])
                val_loss += loss.item()
            
            val_loss /= (len(loaders['val']) * args['batch'])
            print("val loss = {:.6f}".format(val_loss))
        
            model.train()
        
        if args['save_latent_gif']:
            save_latents_as_gif(args, out, epoch)

        # Log results in wandb
        if wandb is not None:
            wandb.log({"train-loss": train_loss, "val-loss": val_loss})

    if wandb is not None:
        run.finish()
    
    torch.save(model, args['model_savename'])
    
    return train_loss, val_loss, losses, args



def test(args, model, device, loaders):

    # Evaluate on testing dataset
    with torch.no_grad():

        model.eval()
        test_loss = 0

        p = torch.Tensor([args['p']]).to(device)

        for data in loaders['test']:

            data = data.to(device)
            out = model(data, epoch=-1)
            loss, _ = total_loss(out, p, args['beta_z'], args['beta_d'])
            test_loss += loss.item()

            if args['save_latents']:
                save_latent(out['z_2'].to('cpu').numpy(), args['latent_filename'])

    # Do interpretability stuff
    neuron_attribution(data, model)

    test_loss /= (len(loaders['test']) * args['batch'])
    print("Test loss = {:.6f}".format(test_loss))



if __name__=='__main__':

    # Track loss terms for plotting
    losses = {'z0_prior' : [],
              'z1_prior' : [],
              'l0_l3' : [],
              'l1_l2' : [],
              'a_app' : [],
              'a_reg' : [],
              'x0_recon' : [],
              'x1_recon' : [],
              'x0_x3' : [],
              'x1_x2' : []}


    # Load args from configs.conf
    args = load_args('configs.conf')

    # If usecuda, operations will be performed on GPU
    usecuda = torch.cuda.is_available() and not args['no_cuda']
    loaders = load_data(**args, usecuda=usecuda)

    device = torch.device("cuda") if usecuda else torch.device("cpu")
    print("Using device", device)

    # Create model
    if args['load_model'] is not None:
        model = torch.load(args['load_model']).to(device)
    else:
        model = Model(**args, device=device).to(device)

    train_loss, val_loss, losses, args = train()


    save_loss_plots(losses, args['beta_d'], args['beta_z'], args['fluents'])
    print("Beta_d = %d, beta_z = %d, fluents = %d finished with train loss %.5f and val loss %.5f" % (args['beta_d'], args['beta_z'], args['fluents'], train_loss, val_loss))