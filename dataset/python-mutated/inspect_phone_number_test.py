import os
import inspect_phone_number as inspect_content
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_inspect_phone_number(capsys: pytest.CaptureFixture) -> None:
    if False:
        return 10
    test_string = 'String with a phone number: 234-555-6789'
    inspect_content.inspect_phone_number(GCLOUD_PROJECT, test_string)
    (out, _) = capsys.readouterr()
    assert 'Info type: PHONE_NUMBER' in out
    assert 'Quote: 234-555-6789' in out