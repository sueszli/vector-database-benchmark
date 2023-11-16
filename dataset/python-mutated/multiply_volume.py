import numpy as np
from moviepy.decorators import audio_video_fx, convert_parameter_to_seconds

def _multiply_volume_in_range(factor, start_time, end_time, nchannels):
    if False:
        while True:
            i = 10

    def factors_filter(factor, t):
        if False:
            print('Hello World!')
        return np.array([factor if start_time <= t_ <= end_time else 1 for t_ in t])

    def multiply_stereo_volume(get_frame, t):
        if False:
            i = 10
            return i + 15
        return np.multiply(get_frame(t), np.array([factors_filter(factor, t) for _ in range(nchannels)]).T)

    def multiply_mono_volume(get_frame, t):
        if False:
            i = 10
            return i + 15
        return np.multiply(get_frame(t), factors_filter(factor, t))
    return multiply_mono_volume if nchannels == 1 else multiply_stereo_volume

@audio_video_fx
@convert_parameter_to_seconds(['start_time', 'end_time'])
def multiply_volume(clip, factor, start_time=None, end_time=None):
    if False:
        for i in range(10):
            print('nop')
    "Returns a clip with audio volume multiplied by the\n    value `factor`. Can be applied to both audio and video clips.\n\n    Parameters\n    ----------\n\n    factor : float\n      Volume multiplication factor.\n\n    start_time : float, optional\n      Time from the beginning of the clip until the volume transformation\n      begins to take effect, in seconds. By default at the beginning.\n\n    end_time : float, optional\n      Time from the beginning of the clip until the volume transformation\n      ends to take effect, in seconds. By default at the end.\n\n    Examples\n    --------\n\n    >>> from moviepy import AudioFileClip\n    >>>\n    >>> music = AudioFileClip('music.ogg')\n    >>> doubled_audio_clip = clip.multiply_volume(2)  # doubles audio volume\n    >>> half_audio_clip = clip.multiply_volume(0.5)  # half audio\n    >>>\n    >>> # silenced clip during one second at third\n    >>> silenced_clip = clip.multiply_volume(0, start_time=2, end_time=3)\n    "
    if start_time is None and end_time is None:
        return clip.transform(lambda get_frame, t: factor * get_frame(t), keep_duration=True)
    return clip.transform(_multiply_volume_in_range(factor, clip.start if start_time is None else start_time, clip.end if end_time is None else end_time, clip.nchannels), keep_duration=True)