import os
import deidentify_redact as deid
import pytest
HARMFUL_STRING = 'My SSN is 372819127'
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_deidentify_with_redact(capsys: pytest.CaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    deid.deidentify_with_redact(GCLOUD_PROJECT, HARMFUL_STRING + '!', ['US_SOCIAL_SECURITY_NUMBER'])
    (out, _) = capsys.readouterr()
    assert 'My SSN is !' in out