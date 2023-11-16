import numpy as np
from moviepy.video.VideoClip import ImageClip

def mask_or(clip, other_clip):
    if False:
        for i in range(10):
            print('nop')
    "Returns the logical 'or' (maximum pixel color values) between two masks.\n\n    The result has the duration of the clip to which has been applied, if it has any.\n\n    Parameters\n    ----------\n\n    other_clip ImageClip or np.ndarray\n      Clip used to mask the original clip.\n\n    Examples\n    --------\n\n    >>> clip = ColorClip(color=(255, 0, 0), size=(1, 1))  # red\n    >>> mask = ColorClip(color=(0, 255, 0), size=(1, 1))  # green\n    >>> masked_clip = clip.fx(mask_or, mask)              # yellow\n    >>> masked_clip.get_frame(0)\n    [[[255 255   0]]]\n    "
    if isinstance(other_clip, ImageClip):
        other_clip = other_clip.img
    if isinstance(other_clip, np.ndarray):
        return clip.image_transform(lambda frame: np.maximum(frame, other_clip))
    else:
        return clip.transform(lambda get_frame, t: np.maximum(get_frame(t), other_clip.get_frame(t)))