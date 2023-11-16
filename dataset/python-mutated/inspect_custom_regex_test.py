import os
import inspect_custom_regex as custom_infotype
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_inspect_data_with_custom_regex_detector(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    custom_infotype.inspect_data_with_custom_regex_detector(GCLOUD_PROJECT, 'Patients MRN 444-5-22222')
    (out, _) = capsys.readouterr()
    assert 'Info type: C_MRN' in out