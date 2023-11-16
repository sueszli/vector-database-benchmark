from moviepy.audio.AudioClip import concatenate_audioclips
from moviepy.decorators import audio_video_fx

@audio_video_fx
def audio_loop(clip, n_loops=None, duration=None):
    if False:
        i = 10
        return i + 15
    "Loops over an audio clip.\n\n    Returns an audio clip that plays the given clip either\n    `n_loops` times, or during `duration` seconds.\n\n    Examples\n    --------\n\n    >>> from moviepy import *\n    >>> videoclip = VideoFileClip('myvideo.mp4')\n    >>> music = AudioFileClip('music.ogg')\n    >>> audio = afx.audio_loop( music, duration=videoclip.duration)\n    >>> videoclip.with_audio(audio)\n\n    "
    if duration is not None:
        n_loops = int(duration / clip.duration) + 1
        return concatenate_audioclips(n_loops * [clip]).with_duration(duration)
    return concatenate_audioclips(n_loops * [clip])