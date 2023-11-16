import os
import deidentify_dictionary_replacement as deid
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_deindentify_with_dictionary_replacement(capsys: pytest.CaptureFixture) -> None:
    if False:
        return 10
    deid.deindentify_with_dictionary_replacement(GCLOUD_PROJECT, 'My name is Alicia Abernathy, and my email address is aabernathy@example.com.', ['EMAIL_ADDRESS'], ['izumi@example.com', 'alex@example.com', 'tal@example.com'])
    (out, _) = capsys.readouterr()
    assert 'aabernathy@example.com' not in out
    assert 'izumi@example.com' in out or 'alex@example.com' in out or 'tal@example.com' in out