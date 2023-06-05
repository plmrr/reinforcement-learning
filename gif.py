import os
from PIL import Image


def create_gif(image_folder, output_file, duration=500):
    images = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            filepath = os.path.join(image_folder, filename)
            try:
                image = Image.open(filepath)
                images.append(image)
            except IOError:
                pass

    images[0].save(output_file, format='GIF',
                   append_images=images[1:],
                   save_all=True,
                   duration=duration,
                   loop=0)
