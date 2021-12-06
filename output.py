from PIL import Image
import matplotlib.pyplot as plt

def save_as_gif(result, filepath):

    print("There are %d images" % (len(result)))
    gif = []
    for arr in result:
        gif.append(Image.fromarray((arr[0] + 1) * 127.5))

    gif[0].save(filepath, save_all=True, append_images=gif[1:], duration=1000, loop=0)


def save_image(result, filepath):

    img = Image.fromarray((result[0] + 1) * 127.5).convert('RGB')
    img.save(filepath)


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