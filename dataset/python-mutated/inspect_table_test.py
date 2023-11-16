import os
import inspect_table as inspect_content
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_inspect_table(capsys: pytest.CaptureFixture) -> None:
    if False:
        i = 10
        return i + 15
    test_tabular_data = {'header': ['email', 'phone number'], 'rows': [['robertfrost@xyz.com', '4232342345'], ['johndoe@pqr.com', '4253458383']]}
    inspect_content.inspect_table(GCLOUD_PROJECT, test_tabular_data, ['PHONE_NUMBER', 'EMAIL_ADDRESS'], include_quote=True)
    (out, _) = capsys.readouterr()
    assert 'Info type: PHONE_NUMBER' in out
    assert 'Info type: EMAIL_ADDRESS' in out