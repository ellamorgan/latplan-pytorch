import numpy as np
from PIL import Image


def load_puzzle(w, n_data, n_tiles=3, image='cat', train_split=0.8, val_split=0.1):
    """
    Loads the puzzle domain
    Returns data normalized to range [-1, 1] in format (n_data * 2, 1, w, w)
    """
    with np.load('data/puzzle-' + str(n_tiles) + '.npz') as data:
        pres = data['pres'].argsort()[:n_data]
        sucs = data['sucs'].argsort()[:n_data]
    
    n_data = len(pres)
    image = Image.open('data/images/' + image + '.jpg').resize((w, w)).convert('L')
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

    train_ind = int(len(data) * train_split)
    val_ind = int(len(data) * (train_split + val_split))

    train_data = data[:train_ind]
    val_data = data[train_ind:val_ind]
    test_data = data[val_ind:]

    img_save_filepath = 'data/puzzle-images-' + str(n_tiles) + '-' + image + '.npz'
    np.savez(img_save_filepath, pres = pres_images, sucs = sucs_images)

    print("Puzzle data loaded in with shape", data.shape)


def load_lightsout():
    pass


def load_hanoi():
    pass


if __name__ == '__main__':
    load_puzzle(w=60, n_data=1000, n_tiles=5, image_path='data/images/cat.jpg')