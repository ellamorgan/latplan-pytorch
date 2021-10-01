import argparse
import numpy as np
from PIL import Image
import pickle


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


def load_cifar10():
    """
    Load cifar10 to test autoencoder
    """
    with open('data/cifar-10/data_batch_1', 'rb') as fo:
        X_dict = pickle.load(fo, encoding='bytes')
    X_train = X_dict[b'data']           # shape: (10000, 3072) = (10000 images, 32x32 r + 32x32 b + 32x32 g)
    y_train = X_dict[b'labels']         # list of length 10000

    print("X_train shape: ", end="")
    print(X_train.shape)
    print("Length of y_train: ", end="")
    print(len(y_train))
    return None



def load_data(dataset, usecuda=False):
    """
    Loads data, returns training, validation, and testing data loader
    """

    if dataset == 'puzzle':
        data = load_puzzle()
    elif dataset == 'cifar10':
        data = load_cifar10()