import os
import pytest
import video_detect_logo_gcs
RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_detect_logo_gcs(capsys):
    if False:
        while True:
            i = 10
    input_uri = 'gs://cloud-samples-data/video/googlework_tiny.mp4'
    video_detect_logo_gcs.detect_logo_gcs(input_uri=input_uri)
    (out, _) = capsys.readouterr()
    assert 'Description' in out
    assert 'Confidence' in out
    assert 'Start Time Offset' in out
    assert 'End Time Offset' in out