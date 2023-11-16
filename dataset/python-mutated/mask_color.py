import numpy as np

def mask_color(clip, color=None, threshold=0, stiffness=1):
    if False:
        return 10
    'Returns a new clip with a mask for transparency where the original\n    clip is of the given color.\n\n    You can also have a "progressive" mask by specifying a non-null distance\n    threshold ``threshold``. In this case, if the distance between a pixel and\n    the given color is d, the transparency will be\n\n    d**stiffness / (threshold**stiffness + d**stiffness)\n\n    which is 1 when d>>threshold and 0 for d<<threshold, the stiffness of the\n    effect being parametrized by ``stiffness``\n    '
    if color is None:
        color = [0, 0, 0]
    color = np.array(color)

    def hill(x):
        if False:
            print('Hello World!')
        if threshold:
            return x ** stiffness / (threshold ** stiffness + x ** stiffness)
        else:
            return 1.0 * (x != 0)

    def flim(im):
        if False:
            print('Hello World!')
        return hill(np.sqrt(((im - color) ** 2).sum(axis=2)))
    mask = clip.image_transform(flim)
    mask.is_mask = True
    new_clip = clip.with_mask(mask)
    return new_clip