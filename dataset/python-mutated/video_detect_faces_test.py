import os
import pytest
import video_detect_faces
RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_detect_faces(capsys):
    if False:
        i = 10
        return i + 15
    local_file_path = os.path.join(RESOURCES, 'googlework_short.mp4')
    video_detect_faces.detect_faces(local_file_path=local_file_path)
    (out, _) = capsys.readouterr()
    assert 'Face detected:' in out