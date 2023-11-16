import moviepy.video.compositing.transitions as transfx
from moviepy.decorators import requires_duration
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

@requires_duration
def make_loopable(clip, overlap_duration):
    if False:
        print('Hello World!')
    'Makes the clip fade in progressively at its own end, this way it can be\n    looped indefinitely.\n\n    Parameters\n    ----------\n\n    overlap_duration : float\n      Duration of the fade-in (in seconds).\n    '
    clip2 = clip.fx(transfx.crossfadein, overlap_duration).with_start(clip.duration - overlap_duration)
    return CompositeVideoClip([clip, clip2]).subclip(overlap_duration, clip.duration)