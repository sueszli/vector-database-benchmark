import os
import deidentify_exception_list as deid
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_deidentify_with_exception_list(capsys: pytest.CaptureFixture) -> None:
    if False:
        i = 10
        return i + 15
    content_str = 'jack@example.org accessed record of user: gary@example.org'
    exception_list = ['jack@example.org', 'jill@example.org']
    deid.deidentify_with_exception_list(GCLOUD_PROJECT, content_str, ['EMAIL_ADDRESS'], exception_list)
    (out, _) = capsys.readouterr()
    assert 'gary@example.org' not in out
    assert 'jack@example.org accessed record of user: [EMAIL_ADDRESS]' in out