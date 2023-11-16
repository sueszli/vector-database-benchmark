from moviepy.decorators import audio_video_fx

@audio_video_fx
def multiply_stereo_volume(clip, left=1, right=1):
    if False:
        while True:
            i = 10
    "For a stereo audioclip, this function enables to change the volume\n    of the left and right channel separately (with the factors `left`\n    and `right`). Makes a stereo audio clip in which the volume of left\n    and right is controllable.\n\n    Examples\n    --------\n\n    >>> from moviepy import AudioFileClip\n    >>> music = AudioFileClip('music.ogg')\n    >>> audio_r = music.multiply_stereo_volume(left=0, right=1)  # mute left channel/s\n    >>> audio_h = music.multiply_stereo_volume(left=0.5, right=0.5)  # half audio\n    "

    def stereo_volume(get_frame, t):
        if False:
            while True:
                i = 10
        frame = get_frame(t)
        if len(frame) == 1:
            frame *= left if left is not None else right
        else:
            for i in range(len(frame[0])):
                frame[:, i] *= left if i % 2 == 0 else right
        return frame
    return clip.transform(stereo_volume, keep_duration=True)