import os
import inspect_string_with_exclusion_regex as custom_infotype
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_inspect_string_with_exclusion_regex(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        return 10
    custom_infotype.inspect_string_with_exclusion_regex(GCLOUD_PROJECT, 'alice@example.com, ironman@avengers.net', '.+@example.com')
    (out, _) = capsys.readouterr()
    assert 'alice' not in out
    assert 'ironman' in out