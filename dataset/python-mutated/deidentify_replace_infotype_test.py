import os
import deidentify_replace_infotype as deid
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_deidentify_with_replace_infotype(capsys: pytest.CaptureFixture) -> None:
    if False:
        print('Hello World!')
    url_to_redact = 'https://cloud.google.com'
    deid.deidentify_with_replace_infotype(GCLOUD_PROJECT, 'My favorite site is ' + url_to_redact, ['URL'])
    (out, _) = capsys.readouterr()
    assert url_to_redact not in out
    assert 'My favorite site is [URL]' in out