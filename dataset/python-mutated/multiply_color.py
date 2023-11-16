import numpy as np

def multiply_color(clip, factor):
    if False:
        i = 10
        return i + 15
    "\n    Multiplies the clip's colors by the given factor, can be used\n    to decrease or increase the clip's brightness (is that the\n    right word ?)\n    "
    return clip.image_transform(lambda frame: np.minimum(255, factor * frame).astype('uint8'))