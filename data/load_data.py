import argparse
import numpy as np
from PIL import Image


def load_args():
    """
    Parses command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="puzzle")

    # Flags
    parser.add_argument("-no_cuda", action="store_false")
    parser.add_argument("-no_wandb", action="store_false")
    args = parser.parse_args()

    return args



def load_puzzle(n_data = 200, n_tiles = 4, image_name='spider', w = 120, **kwargs):
    """
    Loads the puzzle domain
    """
    with np.load('puzzle-' + n_tiles + '.npz') as data:
        pres_configs = data['pres'][:n_data].argsort()
        sucs_configs = data['sucs'][:n_data].argsort()
    
    image = Image.open('images/spider.png').resize((w, w)).convert('L')
    puzzle = np.asarray(image)
    t_w = w // n_tiles
    tiled_image = np.array([puzzle[i * t_w : (i + 1) * t_w, j * t_w : (j + 1) * t_w] for i in range(n_tiles) for j in range(n_tiles)])




def load_data(dataset, usecuda):
    """
    Loads data, returns training, validation, and testing data loader
    """

    # How to dynamically call functions in a pythonic way?
    # Not creating functions dynamically...
    # Callbacks - a way to stick stuff into the training loop when you don't have access (ex. pytorch lightning, keras)
    # If you have multiple training instances, you want callbacks to know when everything is finished
    # Make this an if statement
    try:
        data_func = globals()["load_" + dataset]
    except KeyError:
        print("Invalid data name")
        exit()