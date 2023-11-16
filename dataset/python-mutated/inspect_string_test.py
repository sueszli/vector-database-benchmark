import os
import inspect_string as inspect_content
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_inspect_string(capsys: pytest.CaptureFixture) -> None:
    if False:
        print('Hello World!')
    test_string = 'My name is Gary Smith and my email is gary@example.com'
    inspect_content.inspect_string(GCLOUD_PROJECT, test_string, ['FIRST_NAME', 'EMAIL_ADDRESS'], include_quote=True)
    (out, _) = capsys.readouterr()
    assert 'Info type: FIRST_NAME' in out
    assert 'Info type: EMAIL_ADDRESS' in out