import re
from google.api_core.retry import Retry
import pytest
import profanity_filter

@Retry()
def test_profanity_filter(capsys: pytest.CaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    result = profanity_filter.sync_recognize_with_profanity_filter_gcs('gs://cloud-samples-tests/speech/brooklyn.flac')
    (out, err) = capsys.readouterr()
    assert re.search('how old is the Brooklyn Bridge', out, re.DOTALL | re.I)
    assert result is not None