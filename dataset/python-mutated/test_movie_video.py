from panda3d.core import MovieVideo
from panda3d.core import Filename
from panda3d.core import PandaSystem
import pytest
import os

def check_ffmpeg():
    if False:
        i = 10
        return i + 15
    MovieVideo.get('test.mp4')
    system = PandaSystem.get_global_ptr()
    return 'FFmpeg' in system.systems

@pytest.mark.skipif(not check_ffmpeg(), reason='skip when ffmpeg is not available')
class Test_Video_Movie:

    def test_cursor_check(self):
        if False:
            i = 10
            return i + 15
        movie_path = os.path.join(os.path.dirname(__file__), 'small.mp4')
        movie_path = Filename.from_os_specific(movie_path)
        reference_file = MovieVideo.get(movie_path)
        assert reference_file.get_filename() == movie_path
        assert reference_file.open() is not None

    def test_video_length(self):
        if False:
            i = 10
            return i + 15
        movie_path = os.path.join(os.path.dirname(__file__), 'small.mp4')
        movie_path = Filename.from_os_specific(movie_path)
        reference_file = MovieVideo.get(movie_path)
        cursor = reference_file.open()
        assert cursor.length() == 32.48

    def test_video_size(self):
        if False:
            return 10
        movie_path = os.path.join(os.path.dirname(__file__), 'small.mp4')
        movie_path = Filename.from_os_specific(movie_path)
        reference_file = MovieVideo.get(movie_path)
        cursor = reference_file.open()
        assert cursor.size_x() == 640
        assert cursor.size_y() == 360