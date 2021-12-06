import torch
import torch.nn as nn
import numpy as np
import itertools

from data.load_data import load_args, load_data
from model import Model
from loss import total_loss
from output import save_as_gif, save_image, save_loss_plots

"""
Flags:
-no_cuda        (train on CPU)
-no_wandb       (don't log metrics on wandb)
"""



def train():

    # Store loss terms for graphing
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

    # Need to put this tensor on the device, done here instead of passing device
    p = torch.Tensor([args['p']]).to(device)

    # Create model
    model = Model(**args, device=device).to(device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3, weight_decay=1e-5)

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
        
        # Save results as gif
        if (epoch + 1) % args['save_every'] == 0:
            
            pres_dec = out['x_dec_0'].to('cpu').numpy()
            sucs_dec = out['x_dec_1'].to('cpu').numpy()
            pres_aae = out['x_aae_3'].to('cpu').numpy()
            sucs_aae = out['x_aae_2'].to('cpu').numpy()

            dec_joint = np.concatenate((pres_dec, sucs_dec), axis=3)
            aae_joint = np.concatenate((pres_aae, sucs_aae), axis=3)
            joint = np.concatenate((dec_joint, aae_joint), axis=2)
            
            save_as_gif(joint, 'saved_gifs/' + str(args['beta_d']) + '_' + str(args['beta_z']) + '_' + str(args['fluents']) + '_' + str(epoch + 1) + '.gif')


        # Log results in wandb
        if wandb is not None:
            wandb.log({"train-loss": train_loss, "val-loss": val_loss})

    if wandb is not None:
        run.finish()
    
    return train_loss, val_loss, losses, args



if __name__=='__main__':

    train_loss, val_loss, losses, args = train()
    save_loss_plots(losses, args['beta_d'], args['beta_z'], args['fluents'])
    print("Beta_d = %d, beta_z = %d, fluents = %d finished with train loss %.5f and val loss %.5f" % (args['beta_d'], args['beta_z'], args['fluents'], train_loss, val_loss))