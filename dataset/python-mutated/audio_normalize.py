from moviepy.audio.fx.multiply_volume import multiply_volume
from moviepy.decorators import audio_video_fx

@audio_video_fx
def audio_normalize(clip):
    if False:
        print('Hello World!')
    "Return a clip whose volume is normalized to 0db.\n\n    Return an audio (or video) clip whose audio volume is normalized\n    so that the maximum volume is at 0db, the maximum achievable volume.\n\n    Examples\n    --------\n\n    >>> from moviepy import *\n    >>> videoclip = VideoFileClip('myvideo.mp4').fx(afx.audio_normalize)\n\n    "
    max_volume = clip.max_volume()
    if max_volume == 0:
        return clip.copy()
    else:
        return multiply_volume(clip, 1 / max_volume)