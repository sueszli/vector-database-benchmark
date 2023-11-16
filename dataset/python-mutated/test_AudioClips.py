"""Image sequencing clip tests meant to be run with pytest."""
import os
import numpy as np
import pytest
from moviepy.audio.AudioClip import AudioArrayClip, AudioClip, CompositeAudioClip, concatenate_audioclips
from moviepy.audio.io.AudioFileClip import AudioFileClip

def test_audioclip(util, mono_wave):
    if False:
        while True:
            i = 10
    filename = os.path.join(util.TMP_DIR, 'audioclip.mp3')
    audio = AudioClip(mono_wave(440), duration=2, fps=22050)
    audio.write_audiofile(filename, bitrate='16', logger=None)
    assert os.path.exists(filename)
    AudioFileClip(filename)

def test_audioclip_io(util):
    if False:
        while True:
            i = 10
    filename = os.path.join(util.TMP_DIR, 'random.wav')
    input_array = np.random.random((220000, 2)) * 1.98 - 0.99
    clip = AudioArrayClip(input_array, fps=44100)
    clip.write_audiofile(filename, logger=None)
    clip = AudioFileClip(filename)
    output_array = clip.to_soundarray()
    np.testing.assert_array_almost_equal(output_array[:len(input_array)], input_array, decimal=4)
    assert (output_array[len(input_array):] == 0).all()

def test_concatenate_audioclips_render(util, mono_wave):
    if False:
        print('Hello World!')
    'Concatenated AudioClips through ``concatenate_audioclips`` should return\n    a clip that can be rendered to a file.\n    '
    filename = os.path.join(util.TMP_DIR, 'concatenate_audioclips.mp3')
    clip_440 = AudioClip(mono_wave(440), duration=0.01, fps=44100)
    clip_880 = AudioClip(mono_wave(880), duration=1e-06, fps=22050)
    concat_clip = concatenate_audioclips((clip_440, clip_880))
    concat_clip.write_audiofile(filename, logger=None)
    assert concat_clip.duration == clip_440.duration + clip_880.duration

def test_concatenate_audioclips_CompositeAudioClip():
    if False:
        print('Hello World!')
    'Concatenated AudioClips through ``concatenate_audioclips`` should return\n    a CompositeAudioClip whose attributes should be consistent:\n\n    - Returns CompositeAudioClip.\n    - Their fps is taken from the maximum of their audios.\n    - Audios are placed one after other:\n      - Duration is the sum of their durations.\n      - Ends are the accumulated sum of their durations.\n      - Starts are the accumulated sum of their durations, but first start is 0\n      and latest is ignored.\n    - Channels are the max channels of their clips.\n    '
    frequencies = [440, 880, 1760]
    durations = [2, 5, 1]
    fpss = [44100, 22050, 11025]
    clips = [AudioClip(lambda t: [np.sin(frequency * 2 * np.pi * t)], duration=duration, fps=fps) for (frequency, duration, fps) in zip(frequencies, durations, fpss)]
    concat_clip = concatenate_audioclips(clips)
    assert isinstance(concat_clip, CompositeAudioClip)
    assert concat_clip.fps == 44100
    assert concat_clip.duration == sum(durations)
    assert list(concat_clip.ends) == list(np.cumsum(durations))
    assert list(concat_clip.starts), list(np.cumsum([0, *durations[:-1]]))
    assert concat_clip.nchannels == max((clip.nchannels for clip in clips))

def test_CompositeAudioClip_by__init__():
    if False:
        return 10
    'The difference between the CompositeAudioClip returned by\n    ``concatenate_audioclips`` and a CompositeAudioClip created using the class\n    directly, is that audios in ``concatenate_audioclips`` are played one after\n    other and AudioClips passed to CompositeAudioClip can be played at different\n    times, it depends on their ``start`` attributes.\n    '
    frequencies = [440, 880, 1760]
    durations = [2, 5, 1]
    fpss = [44100, 22050, 11025]
    starts = [0, 1, 2]
    clips = [AudioClip(lambda t: [np.sin(frequency * 2 * np.pi * t)], duration=duration, fps=fps).with_start(start) for (frequency, duration, fps, start) in zip(frequencies, durations, fpss, starts)]
    compound_clip = CompositeAudioClip(clips)
    assert isinstance(compound_clip, CompositeAudioClip)
    assert compound_clip.fps == 44100
    ends = [start + duration for (start, duration) in zip(starts, durations)]
    assert compound_clip.duration == max(ends)
    assert list(compound_clip.ends) == ends
    assert list(compound_clip.starts) == starts
    assert compound_clip.nchannels == max((clip.nchannels for clip in clips))

def test_concatenate_audioclip_with_audiofileclip(util, stereo_wave):
    if False:
        print('Hello World!')
    clip1 = AudioClip(stereo_wave(left_freq=440, right_freq=880), duration=1, fps=44100)
    clip2 = AudioFileClip('media/crunching.mp3')
    concat_clip = concatenate_audioclips((clip1, clip2))
    concat_clip.write_audiofile(os.path.join(util.TMP_DIR, 'concat_clip_with_file_audio.mp3'), logger=None)
    assert concat_clip.duration == clip1.duration + clip2.duration

def test_concatenate_audiofileclips(util):
    if False:
        while True:
            i = 10
    clip1 = AudioFileClip('media/crunching.mp3').subclip(1, 4)
    clip2 = AudioFileClip('media/big_buck_bunny_432_433.webm')
    concat_clip = concatenate_audioclips((clip1, clip2))
    concat_clip.write_audiofile(os.path.join(util.TMP_DIR, 'concat_audio_file.mp3'), logger=None)
    assert concat_clip.duration == clip1.duration + clip2.duration

def test_audioclip_mono_max_volume(mono_wave):
    if False:
        print('Hello World!')
    clip = AudioClip(mono_wave(440), duration=1, fps=44100)
    max_volume = clip.max_volume()
    assert isinstance(max_volume, float)
    assert max_volume > 0

@pytest.mark.parametrize('nchannels', (2, 4, 8, 16))
@pytest.mark.parametrize('channel_muted', ('left', 'right'))
def test_audioclip_stereo_max_volume(nchannels, channel_muted):
    if False:
        for i in range(10):
            print('nop')

    def make_frame(t):
        if False:
            i = 10
            return i + 15
        frame = []
        for i in range(int(nchannels / 2)):
            if channel_muted == 'left':
                frame.append(np.sin(t * 0))
                frame.append(np.sin(440 * 2 * np.pi * t))
            else:
                frame.append(np.sin(440 * 2 * np.pi * t))
                frame.append(np.sin(t * 0))
        return np.array(frame).T
    clip = AudioClip(make_frame, fps=44100, duration=1)
    max_volume = clip.max_volume(stereo=True)
    assert isinstance(max_volume, np.ndarray)
    assert len(max_volume) == nchannels
    for (i, channel_max_volume) in enumerate(max_volume):
        if i % 2 == 0:
            if channel_muted == 'left':
                assert channel_max_volume == 0
            else:
                assert channel_max_volume > 0
        elif channel_muted == 'right':
            assert channel_max_volume == 0
        else:
            assert channel_max_volume > 0
if __name__ == '__main__':
    pytest.main()