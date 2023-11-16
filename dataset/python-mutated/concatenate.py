"""Video clips concatenation."""
from functools import reduce
import numpy as np
from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.VideoClip import ColorClip, VideoClip

def concatenate_videoclips(clips, method='chain', transition=None, bg_color=None, is_mask=False, padding=0):
    if False:
        i = 10
        return i + 15
    'Concatenates several video clips.\n\n    Returns a video clip made by clip by concatenating several video clips.\n    (Concatenated means that they will be played one after another).\n\n    There are two methods:\n\n    - method="chain": will produce a clip that simply outputs\n      the frames of the successive clips, without any correction if they are\n      not of the same size of anything. If none of the clips have masks the\n      resulting clip has no mask, else the mask is a concatenation of masks\n      (using completely opaque for clips that don\'t have masks, obviously).\n      If you have clips of different size and you want to write directly the\n      result of the concatenation to a file, use the method "compose" instead.\n\n    - method="compose", if the clips do not have the same resolution, the final\n      resolution will be such that no clip has to be resized.\n      As a consequence the final clip has the height of the highest clip and the\n      width of the widest clip of the list. All the clips with smaller dimensions\n      will appear centered. The border will be transparent if mask=True, else it\n      will be of the color specified by ``bg_color``.\n\n    The clip with the highest FPS will be the FPS of the result clip.\n\n    Parameters\n    ----------\n    clips\n      A list of video clips which must all have their ``duration``\n      attributes set.\n    method\n      "chain" or "compose": see above.\n    transition\n      A clip that will be played between each two clips of the list.\n\n    bg_color\n      Only for method=\'compose\'. Color of the background.\n      Set to None for a transparent clip\n\n    padding\n      Only for method=\'compose\'. Duration during two consecutive clips.\n      Note that for negative padding, a clip will partly play at the same\n      time as the clip it follows (negative padding is cool for clips who fade\n      in on one another). A non-null padding automatically sets the method to\n      `compose`.\n\n    '
    if transition is not None:
        clip_transition_pairs = [[v, transition] for v in clips[:-1]]
        clips = reduce(lambda x, y: x + y, clip_transition_pairs) + [clips[-1]]
        transition = None
    timings = np.cumsum([0] + [clip.duration for clip in clips])
    sizes = [clip.size for clip in clips]
    w = max((size[0] for size in sizes))
    h = max((size[1] for size in sizes))
    timings = np.maximum(0, timings + padding * np.arange(len(timings)))
    timings[-1] -= padding
    if method == 'chain':

        def make_frame(t):
            if False:
                for i in range(10):
                    print('nop')
            i = max([i for (i, e) in enumerate(timings) if e <= t])
            return clips[i].get_frame(t - timings[i])

        def get_mask(clip):
            if False:
                for i in range(10):
                    print('nop')
            mask = clip.mask or ColorClip([1, 1], color=1, is_mask=True)
            if mask.duration is None:
                mask.duration = clip.duration
            return mask
        result = VideoClip(is_mask=is_mask, make_frame=make_frame)
        if any([clip.mask is not None for clip in clips]):
            masks = [get_mask(clip) for clip in clips]
            result.mask = concatenate_videoclips(masks, method='chain', is_mask=True)
            result.clips = clips
    elif method == 'compose':
        result = CompositeVideoClip([clip.with_start(t).with_position('center') for (clip, t) in zip(clips, timings)], size=(w, h), bg_color=bg_color, is_mask=is_mask)
    else:
        raise Exception("MoviePy Error: The 'method' argument of concatenate_videoclips must be 'chain' or 'compose'")
    result.timings = timings
    result.start_times = timings[:-1]
    (result.start, result.duration, result.end) = (0, timings[-1], timings[-1])
    audio_t = [(clip.audio, t) for (clip, t) in zip(clips, timings) if clip.audio is not None]
    if audio_t:
        result.audio = CompositeAudioClip([a.with_start(t) for (a, t) in audio_t])
    fpss = [clip.fps for clip in clips if getattr(clip, 'fps', None) is not None]
    result.fps = max(fpss) if fpss else None
    return result