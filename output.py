import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


def save_as_gif(result, filepath):

    print("There are %d images" % (len(result)))
    gif = []
    for arr in result:
        gif.append(Image.fromarray((arr[0] + 1) * 127.5))

    gif[0].save(filepath, save_all=True, append_images=gif[1:], duration=1000, loop=0)


def save_image(result, filepath):

    img = Image.fromarray((result[0] + 1) * 127.5).convert('RGB')
    img.save(filepath)


def save_latents_as_gif(args, out, epoch):

    # Save results of last batch of validation data as gif
    if (epoch + 1) % args['save_every'] == 0:
        
        pres_dec = out['x_dec_0'].to('cpu').numpy()
        sucs_dec = out['x_dec_1'].to('cpu').numpy()
        pres_aae = out['x_aae_3'].to('cpu').numpy()
        sucs_aae = out['x_aae_2'].to('cpu').numpy()

        dec_joint = np.concatenate((pres_dec, sucs_dec), axis=3)
        aae_joint = np.concatenate((pres_aae, sucs_aae), axis=3)
        joint = np.concatenate((dec_joint, aae_joint), axis=2)
        
        save_as_gif(joint, 'saved_gifs/' + str(args['beta_d']) + '_' + str(args['beta_z']) + '_' + str(args['fluents']) + '_' + str(epoch + 1) + '.gif')


def save_heatmap(heatmap, name):
    '''
    Give an (n, n) matrix and a name, saves as a heatmap
    '''
    fig, ax = plt.subplots(figsize=(12,7))
    plt.title('Variable densities', fontsize=18)
    ax.title.set_position([0.5, 1.05])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    sns.heatmap(heatmap, ax=ax, square=True, center=0, vmin=-1, vmax=1)
    plt.savefig(name)
    plt.close(fig)


def save_loss_plots(losses, beta_d, beta_z, fluents):

    plt.plot(range(len(losses['z0_prior'])), losses['z0_prior'], label = 'z0_prior', color = 'mediumseagreen')
    plt.plot(range(len(losses['z1_prior'])), losses['z1_prior'], label = 'z1_prior', color = 'mediumturquoise')
    plt.plot(range(len(losses['a_app'])), losses['a_app'], label = 'a_app', color = 'maroon')
    plt.plot(range(len(losses['a_reg'])), losses['a_reg'], label = 'a_reg', color = 'tomato')
    plt.plot(range(len(losses['x0_recon'])), losses['x0_recon'], label = 'x0_recon', color = 'darkorange')
    plt.plot(range(len(losses['x1_recon'])), losses['x1_recon'], label = 'x1_recon', color = 'gold')
    plt.plot(range(len(losses['x0_x3'])), losses['x0_x3'], label = 'x0_x3', color = 'violet')
    plt.plot(range(len(losses['x1_x2'])), losses['x1_x2'], label = 'x1_x2', color = 'crimson')
    plt.plot(range(len(losses['l0_l3'])), losses['l0_l3'], label = 'l0_l3', color = 'blueviolet')
    plt.plot(range(len(losses['l1_l2'])), losses['l1_l2'], label = 'l1_l2', color = 'mediumslateblue')
    plt.legend()
    plt.savefig('losses/losses_p0.5_' + str(beta_d) + '_' + str(beta_z) + '_' + str(fluents) + '.png')

    plt.clf()
    plt.plot(range(len(losses['z0_prior'])), losses['z0_prior'], label = 'z0_prior', color = 'mediumseagreen')
    plt.plot(range(len(losses['z1_prior'])), losses['z1_prior'], label = 'z1_prior', color = 'mediumturquoise')
    plt.legend()
    plt.savefig('losses/prior_losses_p0.5_' + str(beta_d) + '_' + str(beta_z) + '_' + str(fluents) + '.png')

    plt.clf()
    plt.plot(range(len(losses['l0_l3'])), losses['l0_l3'], label = 'l0_l3', color = 'blueviolet')
    plt.plot(range(len(losses['l1_l2'])), losses['l1_l2'], label = 'l1_l2', color = 'mediumslateblue')
    plt.legend()
    plt.savefig('losses/bc_losses_p0.5_' + str(beta_d) + '_' + str(beta_z) + '_' + str(fluents) + '.png')

    plt.clf()
    plt.plot(range(len(losses['a_app'])), losses['a_app'], label = 'a_app', color = 'maroon')
    plt.plot(range(len(losses['a_reg'])), losses['a_reg'], label = 'a_reg', color = 'tomato')
    plt.plot(range(len(losses['x0_recon'])), losses['x0_recon'], label = 'x0_recon', color = 'darkorange')
    plt.plot(range(len(losses['x1_recon'])), losses['x1_recon'], label = 'x1_recon', color = 'gold')
    plt.plot(range(len(losses['x0_x3'])), losses['x0_x3'], label = 'x0_x3', color = 'violet')
    plt.plot(range(len(losses['x1_x2'])), losses['x1_x2'], label = 'x1_x2', color = 'crimson')
    plt.legend()
    plt.savefig('losses/recon_losses_p0.5_' + str(beta_d) + '_' + str(beta_z) + '_' + str(fluents) + '.png')