"""Video file clip tests meant to be run with pytest."""
import copy
import os
import pytest
from moviepy.video.compositing.CompositeVideoClip import clips_array
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import ColorClip

def test_setup(util):
    if False:
        i = 10
        return i + 15
    'Test VideoFileClip setup.'
    filename = os.path.join(util.TMP_DIR, 'test.mp4')
    red = ColorClip((256, 200), color=(255, 0, 0))
    green = ColorClip((256, 200), color=(0, 255, 0))
    blue = ColorClip((256, 200), color=(0, 0, 255))
    red.fps = green.fps = blue.fps = 10
    with clips_array([[red, green, blue]]).with_duration(5) as video:
        video.write_videofile(filename, logger=None)
    assert os.path.exists(filename)
    clip = VideoFileClip(filename)
    assert clip.duration == 5
    assert clip.fps == 10
    assert clip.size == [256 * 3, 200]
    assert clip.reader.bitrate == 2

def test_ffmpeg_resizing():
    if False:
        return 10
    'Test FFmpeg resizing, to include downscaling.'
    video_file = 'media/big_buck_bunny_432_433.webm'
    target_resolutions = [(128, 128), (128, None), (None, 128), (None, 256)]
    for target_resolution in target_resolutions:
        video = VideoFileClip(video_file, target_resolution=target_resolution)
        frame = video.get_frame(0)
        for (target, observed) in zip(target_resolution[::-1], frame.shape):
            if target is not None:
                assert target == observed
        video.close()

def test_copied_videofileclip_write_videofile(util):
    if False:
        return 10
    'Check that a copied ``VideoFileClip`` can be renderizable using\n    ``write_videofile``, opened from that render and the new video shares\n    the same data that the original clip.\n    '
    input_video_filepath = 'media/big_buck_bunny_432_433.webm'
    output_video_filepath = os.path.join(util.TMP_DIR, 'copied_videofileclip.mp4')
    clip = VideoFileClip(input_video_filepath).subclip(0, 1)
    copied_clip = clip.copy()
    copied_clip.write_videofile(output_video_filepath)
    assert os.path.exists(output_video_filepath)
    copied_clip_from_file = VideoFileClip(output_video_filepath)
    assert copied_clip.fps == copied_clip_from_file.fps
    assert list(copied_clip.size) == copied_clip_from_file.size
    assert isinstance(copied_clip.reader, type(copied_clip_from_file.reader))

def test_videofileclip_safe_deepcopy(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    'Attempts to do a deepcopy of a VideoFileClip will do a mixed copy,\n    being redirected to ``__copy__`` method of ``VideoClip``, see the\n    documentation of ``VideoFileClip.__deepcopy__`` for more information\n    about this.\n    '
    clip = VideoFileClip('media/chaplin.mp4')

    def fake__copy__():
        if False:
            while True:
                i = 10
        return 'foo'
    monkeypatch.setattr(clip, '__copy__', fake__copy__)
    assert copy.deepcopy(clip) == 'foo'
if __name__ == '__main__':
    pytest.main()