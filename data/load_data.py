import argparse
import configparser
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

'''
Data loading and config loading
'''

def load_args(config_path):
    """
    Parses command line arguments
    Returns a dictionary with command line and config file configs
    Gives priority to command line, can overwrite config file configs
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--load_model", type=str)

    # Flags
    parser.add_argument("-no_cuda", action="store_true")
    parser.add_argument("-no_wandb", action="store_true")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    with open(config_path) as f:
        config.read_file(f)
    configs = dict(config['latplan_configs'])

    def match(x):
        float_rg = "(([+-]?([0-9]*)?[.][0-9]+)|([0-9]+[e][-][0-9]+))"
        if re.match(float_rg):
            return float(x)
        if x.isnumeric():
            return int(x)
        if x == 'True' or x == 'true':
            return True
        if x == 'False' or x == 'false':
            return False
        return x

    configs = {k : match(v) for k, v in configs.items()}

    for k, v in vars(args).items():
        if v is not None:
            configs[k] = v
    
    print("Configs used:", configs)

    return {**dict(configs), **vars(args)}



def load_puzzle(w, n_data, n_tiles=3, image_path='data/images/cat.jpg'):
    """
    Loads the puzzle domain
    Returns data normalized to range [-1, 1] in format (n_data * 2, 1, w, w)
    """
    with np.load('data/puzzle-' + str(n_tiles) + '.npz') as data:
        pres = data['pres'].argsort()[:n_data]
        sucs = data['sucs'].argsort()[:n_data]
    
    n_data = len(pres)
    image = Image.open(image_path).resize((w, w)).convert('L')
    puzzle = np.asarray(image)
    t_w = w // n_tiles
    tiled_image = np.array([puzzle[i * t_w : (i + 1) * t_w, j * t_w : (j + 1) * t_w] for i in range(n_tiles) for j in range(n_tiles)])

    def process(permutations):
        d = (tiled_image[permutations, :, :] / 127.5) - 1
        r1 = np.reshape(d, (-1, n_tiles, n_tiles, t_w, t_w))
        t1 = np.transpose(r1, (0, 1, 3, 2, 4))
        r2 = np.reshape(t1, (-1, 1, t_w * n_tiles, t_w * n_tiles))
        return r2

    pres_images = process(pres)
    sucs_images = process(sucs)

    data = np.stack((pres_images, sucs_images), axis=1)

    print("Puzzle data loaded in with shape", data.shape)

    return torch.FloatTensor(data)



def load_data(dataset, img_width, batch, train_split, val_split, n_data, usecuda, **kwargs):
    """
    Loads data, returns training, validation, and testing dataloader
    """
    data = load_puzzle(img_width, n_data)
        
    train_ind = int(len(data) * train_split)
    val_ind = int(len(data) * (train_split + val_split))

    train_data = data[:train_ind]
    val_data = data[train_ind:val_ind]
    test_data = data[val_ind:]
    print("Train / val / test split sizes:", len(train_data), "/", len(val_data), "/", len(test_data))

    train_dataloader = DataLoader(train_data, batch_size=batch, shuffle=True, pin_memory=usecuda, drop_last=True)
    val_dataloader= DataLoader(val_data, batch_size=batch, shuffle=True, pin_memory=usecuda, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batch, shuffle=True, pin_memory=usecuda, drop_last=True)

    return {'train':train_dataloader, 'val':val_dataloader, 'test':test_dataloader}