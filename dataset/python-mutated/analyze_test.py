import time
from google.api_core.exceptions import ServiceUnavailable
import pytest
import analyze
POSSIBLE_TEXTS = ['Google', 'SUR', 'SUR', 'ROTO', 'Vice President', '58oo9', 'LONDRES', 'OMAR', 'PARIS', 'METRO', 'RUE', 'CARLO']

def test_analyze_shots(capsys):
    if False:
        for i in range(10):
            print('nop')
    analyze.analyze_shots('gs://cloud-samples-data/video/gbikes_dinosaur.mp4')
    (out, _) = capsys.readouterr()
    assert 'Shot 1:' in out

def test_analyze_labels(capsys):
    if False:
        while True:
            i = 10
    analyze.analyze_labels('gs://cloud-samples-data/video/cat.mp4')
    (out, _) = capsys.readouterr()
    assert 'label description: cat' in out

def test_analyze_labels_file(capsys):
    if False:
        print('Hello World!')
    analyze.analyze_labels_file('resources/googlework_tiny.mp4')
    (out, _) = capsys.readouterr()
    assert 'label description' in out

def test_analyze_explicit_content(capsys):
    if False:
        print('Hello World!')
    try_count = 0
    while try_count < 3:
        try:
            analyze.analyze_explicit_content('gs://cloud-samples-data/video/cat.mp4')
            (out, _) = capsys.readouterr()
            assert 'pornography' in out
        except ServiceUnavailable as e:
            print('Got service unavailable exception: {}'.format(str(e)))
            time.sleep(5)
            continue
        try_count = try_count + 1
        break

def test_speech_transcription(capsys):
    if False:
        i = 10
        return i + 15
    analyze.speech_transcription('gs://cloud-samples-data/video/googlework_short.mp4')
    (out, _) = capsys.readouterr()
    assert 'cultural' in out

@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_detect_text_gcs(capsys):
    if False:
        for i in range(10):
            print('nop')
    analyze.video_detect_text_gcs('gs://cloud-samples-data/video/googlework_tiny.mp4')
    (out, _) = capsys.readouterr()
    assert 'Text' in out

@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_detect_text(capsys):
    if False:
        print('Hello World!')
    analyze.video_detect_text('resources/googlework_tiny.mp4')
    (out, _) = capsys.readouterr()
    assert 'Text' in out

@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_track_objects_gcs(capsys):
    if False:
        while True:
            i = 10
    analyze.track_objects_gcs('gs://cloud-samples-data/video/cat.mp4')
    (out, _) = capsys.readouterr()
    assert 'cat' in out

@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_track_objects(capsys):
    if False:
        return 10
    in_file = './resources/googlework_tiny.mp4'
    analyze.track_objects(in_file)
    (out, _) = capsys.readouterr()
    assert 'Entity id' in out