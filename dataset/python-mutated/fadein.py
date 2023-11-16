import numpy as np

def fadein(clip, duration, initial_color=None):
    if False:
        i = 10
        return i + 15
    'Makes the clip progressively appear from some color (black by default),\n    over ``duration`` seconds at the beginning of the clip. Can be used for\n    masks too, where the initial color must be a number between 0 and 1.\n\n    For cross-fading (progressive appearance or disappearance of a clip\n    over another clip, see ``transfx.crossfadein``\n    '
    if initial_color is None:
        initial_color = 0 if clip.is_mask else [0, 0, 0]
    initial_color = np.array(initial_color)

    def filter(get_frame, t):
        if False:
            for i in range(10):
                print('nop')
        if t >= duration:
            return get_frame(t)
        else:
            fading = 1.0 * t / duration
            return fading * get_frame(t) + (1 - fading) * initial_color
    return clip.transform(filter)