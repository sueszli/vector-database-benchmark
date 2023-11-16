import os
import pytest
import video_detect_faces_gcs
RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_detect_faces(capsys):
    if False:
        print('Hello World!')
    input_uri = 'gs://cloud-samples-data/video/googlework_short.mp4'
    video_detect_faces_gcs.detect_faces(gcs_uri=input_uri)
    (out, _) = capsys.readouterr()
    assert 'Face detected:' in out