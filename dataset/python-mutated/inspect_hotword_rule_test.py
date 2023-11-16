import os
import inspect_hotword_rule as custom_infotype
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_inspect_with_medical_record_number_w_custom_hotwords_no_hotwords(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        return 10
    custom_infotype.inspect_data_w_custom_hotwords(GCLOUD_PROJECT, 'just a number 444-5-22222')
    (out, _) = capsys.readouterr()
    assert 'Info type: C_MRN' in out
    assert 'Likelihood: 3' in out

def test_inspect_with_medical_record_number_w_custom_hotwords_has_hotwords(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    custom_infotype.inspect_data_w_custom_hotwords(GCLOUD_PROJECT, 'Patients MRN 444-5-22222')
    (out, _) = capsys.readouterr()
    assert 'Info type: C_MRN' in out
    assert 'Likelihood: 5' in out