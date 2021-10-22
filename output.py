from PIL import Image

def save_as_gif(result, filepath):
    gif = []
    for arr in result:
        gif.append(Image.fromarray(arr[0] * 255))
    gif[0].save(filepath, save_all=True, append_images=gif[1:], duration=1000, loop=0)