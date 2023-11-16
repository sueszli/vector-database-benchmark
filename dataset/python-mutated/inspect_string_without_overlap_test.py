import os
import inspect_string_without_overlap as custom_infotype
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_inspect_string_without_overlap(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    custom_infotype.inspect_string_without_overlap(GCLOUD_PROJECT, 'example.com is a domain, james@example.org is an email.')
    (out, _) = capsys.readouterr()
    assert 'example.com' in out
    assert 'example.org' not in out