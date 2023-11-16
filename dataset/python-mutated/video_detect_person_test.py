import os
import pytest
import video_detect_person
RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_detect_person(capsys):
    if False:
        for i in range(10):
            print('nop')
    local_file_path = os.path.join(RESOURCES, 'googlework_tiny.mp4')
    video_detect_person.detect_person(local_file_path=local_file_path)
    (out, _) = capsys.readouterr()
    assert 'Person detected:' in out
    assert 'Attributes:' in out
    assert 'x=' in out
    assert 'y=' in out