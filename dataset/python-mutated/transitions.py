"""Here is the current catalogue. These are meant to be used with ``clip.fx``
There are available as ``transfx.crossfadein`` etc.
"""
from moviepy.decorators import add_mask_if_none, requires_duration
from moviepy.video.fx.fadein import fadein
from moviepy.video.fx.fadeout import fadeout
__all__ = ['crossfadein', 'crossfadeout', 'slide_in', 'slide_out']

@requires_duration
@add_mask_if_none
def crossfadein(clip, duration):
    if False:
        print('Hello World!')
    'Makes the clip appear progressively, over ``duration`` seconds.\n    Only works when the clip is included in a CompositeVideoClip.\n    '
    clip.mask.duration = clip.duration
    new_clip = clip.copy()
    new_clip.mask = clip.mask.fx(fadein, duration)
    return new_clip

@requires_duration
@add_mask_if_none
def crossfadeout(clip, duration):
    if False:
        return 10
    'Makes the clip disappear progressively, over ``duration`` seconds.\n    Only works when the clip is included in a CompositeVideoClip.\n    '
    clip.mask.duration = clip.duration
    new_clip = clip.copy()
    new_clip.mask = clip.mask.fx(fadeout, duration)
    return new_clip

def slide_in(clip, duration, side):
    if False:
        print('Hello World!')
    'Makes the clip arrive from one side of the screen.\n\n    Only works when the clip is included in a CompositeVideoClip,\n    and if the clip has the same size as the whole composition.\n\n    Parameters\n    ----------\n\n    clip : moviepy.Clip.Clip\n      A video clip.\n\n    duration : float\n      Time taken for the clip to be fully visible\n\n    side : str\n      Side of the screen where the clip comes from. One of\n      \'top\', \'bottom\', \'left\' or \'right\'.\n\n    Examples\n    --------\n\n    >>> from moviepy import *\n    >>>\n    >>> clips = [... make a list of clips]\n    >>> slided_clips = [\n    ...     CompositeVideoClip([clip.fx(transfx.slide_in, 1, "left")])\n    ...     for clip in clips\n    ... ]\n    >>> final_clip = concatenate_videoclips(slided_clips, padding=-1)\n    >>>\n    >>> clip = ColorClip(\n    ...     color=(255, 0, 0), duration=1, size=(300, 300)\n    ... ).with_fps(60)\n    >>> final_clip = CompositeVideoClip([transfx.slide_in(clip, 1, "right")])\n    '
    (w, h) = clip.size
    pos_dict = {'left': lambda t: (min(0, w * (t / duration - 1)), 'center'), 'right': lambda t: (max(0, w * (1 - t / duration)), 'center'), 'top': lambda t: ('center', min(0, h * (t / duration - 1))), 'bottom': lambda t: ('center', max(0, h * (1 - t / duration)))}
    return clip.with_position(pos_dict[side])

@requires_duration
def slide_out(clip, duration, side):
    if False:
        return 10
    'Makes the clip go away by one side of the screen.\n\n    Only works when the clip is included in a CompositeVideoClip,\n    and if the clip has the same size as the whole composition.\n\n    Parameters\n    ----------\n\n    clip : moviepy.Clip.Clip\n      A video clip.\n\n    duration : float\n      Time taken for the clip to fully disappear.\n\n    side : str\n      Side of the screen where the clip goes. One of\n      \'top\', \'bottom\', \'left\' or \'right\'.\n\n    Examples\n    --------\n\n    >>> clips = [... make a list of clips]\n    >>> slided_clips = [\n    ...     CompositeVideoClip([clip.fx(transfx.slide_out, 1, "left")])\n    ...     for clip in clips\n    ... ]\n    >>> final_clip = concatenate_videoclips(slided_clips, padding=-1)\n    >>>\n    >>> clip = ColorClip(\n    ...     color=(255, 0, 0), duration=1, size=(300, 300)\n    ... ).with_fps(60)\n    >>> final_clip = CompositeVideoClip([transfx.slide_out(clip, 1, "right")])\n    '
    (w, h) = clip.size
    ts = clip.duration - duration
    pos_dict = {'left': lambda t: (min(0, w * (-(t - ts) / duration)), 'center'), 'right': lambda t: (max(0, w * ((t - ts) / duration)), 'center'), 'top': lambda t: ('center', min(0, h * (-(t - ts) / duration))), 'bottom': lambda t: ('center', max(0, h * ((t - ts) / duration)))}
    return clip.with_position(pos_dict[side])