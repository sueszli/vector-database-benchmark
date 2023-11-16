import os
import deidentify_table_primitive_bucketing as deid
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_deidentify_table_primitive_bucketing(capsys: pytest.CaptureFixture) -> None:
    if False:
        i = 10
        return i + 15
    deid.deidentify_table_primitive_bucketing(GCLOUD_PROJECT)
    (out, _) = capsys.readouterr()
    assert 'string_value: "High"' in out
    assert 'string_value: "Low"' in out