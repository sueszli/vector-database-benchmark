import os
import pytest
import reidentify_free_text_with_fpe_using_surrogate as reid
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
UNWRAPPED_KEY = 'YWJjZGVmZ2hpamtsbW5vcA=='

def test_reidentify_free_text_with_fpe_using_surrogate(capsys: pytest.CaptureFixture) -> None:
    if False:
        print('Hello World!')
    labeled_fpe_string = 'My phone number is PHONE_TOKEN(10):9617256398'
    reid.reidentify_free_text_with_fpe_using_surrogate(GCLOUD_PROJECT, labeled_fpe_string, surrogate_type='PHONE_TOKEN', unwrapped_key=UNWRAPPED_KEY, alphabet='NUMERIC')
    (out, _) = capsys.readouterr()
    assert 'PHONE_TOKEN' not in out
    assert '9617256398' not in out
    assert 'My phone number is' in out