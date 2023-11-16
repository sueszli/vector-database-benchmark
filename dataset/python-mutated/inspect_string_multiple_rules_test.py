import os
import inspect_string_multiple_rules as custom_infotype
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_inspect_string_multiple_rules_patient(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        return 10
    custom_infotype.inspect_string_multiple_rules(GCLOUD_PROJECT, 'patient name: Jane Doe')
    (out, _) = capsys.readouterr()
    assert 'Likelihood: 4' in out

def test_inspect_string_multiple_rules_doctor(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    custom_infotype.inspect_string_multiple_rules(GCLOUD_PROJECT, 'doctor: Jane Doe')
    (out, _) = capsys.readouterr()
    assert 'No findings' in out

def test_inspect_string_multiple_rules_quasimodo(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    custom_infotype.inspect_string_multiple_rules(GCLOUD_PROJECT, 'patient name: quasimodo')
    (out, _) = capsys.readouterr()
    assert 'No findings' in out

def test_inspect_string_multiple_rules_redacted(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    custom_infotype.inspect_string_multiple_rules(GCLOUD_PROJECT, 'name of patient: REDACTED')
    (out, _) = capsys.readouterr()
    assert 'No findings' in out