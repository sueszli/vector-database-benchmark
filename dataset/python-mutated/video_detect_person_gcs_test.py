import os
import pytest
import video_detect_person_gcs
RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_detect_person(capsys):
    if False:
        i = 10
        return i + 15
    input_uri = 'gs://cloud-samples-data/video/googlework_tiny.mp4'
    video_detect_person_gcs.detect_person(gcs_uri=input_uri)
    (out, _) = capsys.readouterr()
    assert 'Person detected:' in out
    assert 'Attributes:' in out
    assert 'x=' in out
    assert 'y=' in out