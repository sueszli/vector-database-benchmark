import os
import inspect_string_custom_excluding_substring as custom_infotype
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_inspect_string_custom_excluding_substring(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    custom_infotype.inspect_string_custom_excluding_substring(GCLOUD_PROJECT, 'Danger, Jimmy | Wayne, Bruce', ['Jimmy'])
    (out, _) = capsys.readouterr()
    assert 'Wayne, Bruce' in out
    assert 'Danger, Jimmy' not in out