import os
import inspect_string_omit_overlap as custom_infotype
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_inspect_string_omit_overlap(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    custom_infotype.inspect_string_omit_overlap(GCLOUD_PROJECT, 'alice@example.com')
    (out, _) = capsys.readouterr()
    assert 'Info type: EMAIL_ADDRESS' in out
    assert 'Info type: PERSON_NAME' not in out