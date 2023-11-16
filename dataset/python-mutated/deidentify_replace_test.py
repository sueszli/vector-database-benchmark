import os
import deidentify_replace as deid
import pytest
HARMFUL_STRING = 'My SSN is 372819127'
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_deidentify_with_replace(capsys: pytest.CaptureFixture) -> None:
    if False:
        while True:
            i = 10
    deid.deidentify_with_replace(GCLOUD_PROJECT, HARMFUL_STRING, ['US_SOCIAL_SECURITY_NUMBER'], replacement_str='REPLACEMENT_STR')
    (out, _) = capsys.readouterr()
    assert 'My SSN is REPLACEMENT_STR' in out