import os
import inspect_string_with_exclusion_dict_substring as custom_infotype
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_inspect_string_with_exclusion_dict_substring(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    custom_infotype.inspect_string_with_exclusion_dict_substring(GCLOUD_PROJECT, 'bob@example.com TEST@example.com TEST.com', ['TEST'])
    (out, _) = capsys.readouterr()
    assert 'TEST@example.com' not in out
    assert 'TEST.com' not in out
    assert 'bob@example.com' in out