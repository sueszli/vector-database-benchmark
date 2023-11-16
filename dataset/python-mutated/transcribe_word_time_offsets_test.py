import os
import re
from google.api_core.retry import Retry
import pytest
import transcribe_word_time_offsets
RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

@Retry()
def test_transcribe_file_with_word_time_offsets(capsys: pytest.CaptureFixture) -> None:
    if False:
        print('Hello World!')
    result = transcribe_word_time_offsets.transcribe_file_with_word_time_offsets(os.path.join(RESOURCES, 'audio.raw'))
    (out, _) = capsys.readouterr()
    print(out)
    match = re.search('Bridge, start_time: ([0-9.]+)', out, re.DOTALL | re.I)
    time = float(match.group(1))
    assert time > 0
    assert result is not None

@Retry()
def test_transcribe_gcs_with_word_time_offsets(capsys: pytest.CaptureFixture) -> None:
    if False:
        while True:
            i = 10
    result = transcribe_word_time_offsets.transcribe_gcs_with_word_time_offsets('gs://python-docs-samples-tests/speech/audio.flac')
    (out, _) = capsys.readouterr()
    print(out)
    match = re.search('Bridge, start_time: ([0-9.]+)', out, re.DOTALL | re.I)
    time = float(match.group(1))
    assert time > 0
    assert result is not None