import os
import inspect_string_custom_hotword as custom_infotype
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_inspect_string_w_custom_hotword(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    custom_infotype.inspect_string_w_custom_hotword(GCLOUD_PROJECT, "patient's name is John Doe.", 'patient')
    (out, _) = capsys.readouterr()
    assert 'Info type: PERSON_NAME' in out
    assert 'Likelihood: 5' in out