"""BitmapClip tests."""
import numpy as np
import pytest
from moviepy.video.VideoClip import BitmapClip

def test_clip_generation():
    if False:
        while True:
            i = 10
    bitmap = [['RR', 'RR'], ['RB', 'RB']]
    expected_frame_array = np.array([np.array([[(255, 0, 0), (255, 0, 0)], [(255, 0, 0), (255, 0, 0)]]), np.array([[(255, 0, 0), (0, 0, 255)], [(255, 0, 0), (0, 0, 255)]])])
    unexpected_frame_array = np.array([np.array([[(255, 0, 0), (255, 0, 0)], [(255, 0, 0), (255, 0, 1)]]), np.array([[(255, 0, 0), (0, 0, 255)], [(255, 0, 0), (0, 0, 255)]])])
    clip = BitmapClip(bitmap, fps=1)
    frame_array = np.array(list(clip.iter_frames()))
    assert np.array_equal(frame_array, expected_frame_array)
    assert not np.array_equal(frame_array, unexpected_frame_array)

def test_setting_fps():
    if False:
        while True:
            i = 10
    bitmap = [['R'], ['R'], ['B'], ['B'], ['G'], ['G']]
    clip = BitmapClip(bitmap, fps=1)
    assert clip.fps == 1
    assert clip.duration == 6

def test_setting_duration():
    if False:
        for i in range(10):
            print('nop')
    bitmap = [['R'], ['R'], ['B'], ['B'], ['G'], ['G']]
    clip = BitmapClip(bitmap, duration=6)
    assert clip.fps == 1
    assert clip.duration == 6

def test_to_bitmap():
    if False:
        print('Hello World!')
    bitmap = [['R'], ['R'], ['B'], ['B'], ['G'], ['G']]
    clip1 = BitmapClip(bitmap, fps=0.345)
    clip2 = BitmapClip(bitmap, fps=1)
    clip3 = BitmapClip(bitmap, fps=3.12345)
    assert bitmap == clip1.to_bitmap()
    assert bitmap == clip2.to_bitmap()
    assert bitmap == clip3.to_bitmap()
if __name__ == '__main__':
    pytest.main()