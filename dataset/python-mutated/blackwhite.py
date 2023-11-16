import numpy as np

def blackwhite(clip, RGB=None, preserve_luminosity=True):
    if False:
        for i in range(10):
            print('nop')
    "Desaturates the picture, makes it black and white.\n    Parameter RGB allows to set weights for the different color\n    channels.\n    If RBG is 'CRT_phosphor' a special set of values is used.\n    preserve_luminosity maintains the sum of RGB to 1.\n    "
    if RGB is None:
        RGB = [1, 1, 1]
    if RGB == 'CRT_phosphor':
        RGB = [0.2125, 0.7154, 0.0721]
    (R, G, B) = 1.0 * np.array(RGB) / (sum(RGB) if preserve_luminosity else 1)

    def filter(im):
        if False:
            i = 10
            return i + 15
        im = R * im[:, :, 0] + G * im[:, :, 1] + B * im[:, :, 2]
        return np.dstack(3 * [im]).astype('uint8')
    return clip.image_transform(filter)