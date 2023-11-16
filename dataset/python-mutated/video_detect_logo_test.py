import os
import pytest
import video_detect_logo
RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_detect_logo(capsys):
    if False:
        print('Hello World!')
    local_file_path = os.path.join(RESOURCES, 'googlework_tiny.mp4')
    video_detect_logo.detect_logo(local_file_path=local_file_path)
    (out, _) = capsys.readouterr()
    assert 'Description' in out
    assert 'Confidence' in out
    assert 'Start Time Offset' in out
    assert 'End Time Offset' in out