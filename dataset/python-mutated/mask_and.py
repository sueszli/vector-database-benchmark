import numpy as np
from moviepy.video.VideoClip import ImageClip

def mask_and(clip, other_clip):
    if False:
        i = 10
        return i + 15
    "Returns the logical 'and' (minimum pixel color values) between two masks.\n\n    The result has the duration of the clip to which has been applied, if it has any.\n\n    Parameters\n    ----------\n\n    other_clip ImageClip or np.ndarray\n      Clip used to mask the original clip.\n\n    Examples\n    --------\n\n    >>> clip = ColorClip(color=(255, 0, 0), size=(1, 1))  # red\n    >>> mask = ColorClip(color=(0, 255, 0), size=(1, 1))  # green\n    >>> masked_clip = clip.fx(mask_and, mask)             # black\n    >>> masked_clip.get_frame(0)\n    [[[0 0 0]]]\n    "
    if isinstance(other_clip, ImageClip):
        other_clip = other_clip.img
    if isinstance(other_clip, np.ndarray):
        return clip.image_transform(lambda frame: np.minimum(frame, other_clip))
    else:
        return clip.transform(lambda get_frame, t: np.minimum(get_frame(t), other_clip.get_frame(t)))