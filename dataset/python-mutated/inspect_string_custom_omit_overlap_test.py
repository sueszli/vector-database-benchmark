import os
import inspect_string_custom_omit_overlap as custom_infotype
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_inspect_string_custom_omit_overlap(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        return 10
    custom_infotype.inspect_string_custom_omit_overlap(GCLOUD_PROJECT, 'Larry Page and John Doe')
    (out, _) = capsys.readouterr()
    assert 'Larry Page' not in out
    assert 'John Doe' in out