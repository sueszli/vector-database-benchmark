from google.api_core.retry import Retry
import pytest
import multi_region

@Retry()
def test_multi_region(capsys: pytest.CaptureFixture) -> None:
    if False:
        i = 10
        return i + 15
    result = multi_region.sync_recognize_with_multi_region_gcs()
    (out, _) = capsys.readouterr()
    assert 'Transcript: how old is the Brooklyn Bridge' in out
    assert result is not None