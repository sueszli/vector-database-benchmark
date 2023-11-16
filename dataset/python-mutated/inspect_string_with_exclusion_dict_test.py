import os
import inspect_string_with_exclusion_dict as custom_infotype
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_inspect_string_with_exclusion_dict(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        return 10
    custom_infotype.inspect_string_with_exclusion_dict(GCLOUD_PROJECT, 'gary@example.com, example@example.com', ['example@example.com'])
    (out, _) = capsys.readouterr()
    assert 'example@example.com' not in out
    assert 'gary@example.com' in out