import numpy as np

from captum.attr import NeuronGradient
from captum.attr import visualization as viz


    
def neuron_attribution(data, model):
    # Layer output is (batch, fluents)
    fluent = 0

    data.requires_grad = True

    # Renormalizes from [-1, 1] to [0, 1], reorders (b, w, h) -> (w, h, b)
    d = data[:, 0].cpu().detach().numpy()
    print("Img shape:", d.shape)
    original_image = np.transpose((d / 2) + 0.5, (0, 2, 3, 1))

    neuron_ig = NeuronGradient(model, model.apply_action)

    model.zero_grad()
    attribution = neuron_ig.attribute(data, (fluent))
    print("Attribution type:", type(attribution))

    attribution = np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0))
    print("Attribution shape:", attribution.shape)

    plt, _ = viz.visualize_image_attr(attribution, original_image, method="blended_heat_map",sign="all",
                                      show_colorbar=True, title="Overlayed Integrated Gradients", use_pyplot=False)
    
    plt.savefig('test.png')



def save_latent(latent, filename):
    # latent: (batch, f)
    with open(filename, 'a') as f:
        for entry in latent:
            f.write(','.join(map(str, entry)) + '\n')


def generate_heatmap(latent_file):
    pass