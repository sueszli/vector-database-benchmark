import os
import deidentify_free_text_with_fpe_using_surrogate as deid
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
UNWRAPPED_KEY = 'YWJjZGVmZ2hpamtsbW5vcA=='

def test_deidentify_free_text_with_fpe_using_surrogate(capsys: pytest.CaptureFixture) -> None:
    if False:
        while True:
            i = 10
    labeled_fpe_string = 'My phone number is 4359916732'
    deid.deidentify_free_text_with_fpe_using_surrogate(GCLOUD_PROJECT, labeled_fpe_string, info_type='PHONE_NUMBER', surrogate_type='PHONE_TOKEN', unwrapped_key=UNWRAPPED_KEY, alphabet='NUMERIC')
    (out, _) = capsys.readouterr()
    assert 'PHONE_TOKEN' in out
    assert 'My phone number is' in out
    assert '4359916732' not in out