import argparse
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader



def load_args():
    """
    Parses command line arguments
    TODO: add config parser with config file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="puzzle")

    # Flags
    parser.add_argument("-no_cuda", action="store_true")
    parser.add_argument("-no_wandb", action="store_true")
    args = parser.parse_args()

    return args



def load_puzzle(n_data = 200, n_tiles = 4, image_path='data/images/spider.png', w = 60):
    """
    Loads the puzzle domain
    Returns data normalized to range [0, 1] in format (n_data * 2, 1, w, w)
    """
    with np.load('data/puzzle-' + str(n_tiles) + '.npz') as data:
        permutations = np.concatenate((data['pres'][:n_data], data['sucs'][:n_data])).argsort()
    
    image = Image.open(image_path).resize((w, w)).convert('L')
    puzzle = np.asarray(image)
    t_w = w // n_tiles
    tiled_image = np.array([puzzle[i * t_w : (i + 1) * t_w, j * t_w : (j + 1) * t_w] for i in range(n_tiles) for j in range(n_tiles)])
    permuted_images = tiled_image[permutations, :, :] / 255
    r1 = np.reshape(permuted_images, (n_data * 2, n_tiles, n_tiles, t_w, t_w))
    t1 = np.transpose(r1, (0, 1, 3, 2, 4))
    r2 = np.reshape(t1, (n_data * 2, w, w))
    data = np.stack((r2[:n_data], r2[n_data:]), axis=1)

    # For now we'll merge the pre and sucs since we're just getting the autoencoder functioning
    data = np.reshape(data, (2 * n_data, 1, w, w))

    print("Puzzle data loaded in with shape", data.shape)

    return torch.FloatTensor(data)



def load_data(dataset, batch_size=20, train_split=0.8, val_split=0.1, usecuda=False, **kwargs):
    """
    Loads data, returns training, validation, and testing data loader
    """
    if dataset == 'puzzle':
        data = load_puzzle(**kwargs)
    
    train_ind = int(len(data) * train_split)
    val_ind = int(len(data) * (train_split + val_split))

    train_data = data[:train_ind]
    val_data = data[train_ind:val_ind]
    test_data = data[val_ind:]
    print("Train / val / test split sizes:", len(train_data), "/", len(val_data), "/", len(test_data))

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=usecuda)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, pin_memory=usecuda)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=usecuda)

    return {'train':train_dataloader, 'val':val_dataloader, 'test':test_dataloader}