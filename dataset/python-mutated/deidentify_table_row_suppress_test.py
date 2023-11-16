import os
import deidentify_table_row_suppress as deid
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
TABLE_DATA = {'header': ['age', 'patient', 'happiness_score'], 'rows': [['101', 'Charles Dickens', '95'], ['22', 'Jane Austen', '21'], ['90', 'Mark Twain', '75']]}

def test_deidentify_table_suppress_row(capsys: pytest.CaptureFixture) -> None:
    if False:
        return 10
    deid.deidentify_table_suppress_row(GCLOUD_PROJECT, TABLE_DATA, 'age', 'GREATER_THAN', 89)
    (out, _) = capsys.readouterr()
    assert 'string_value: "Charles Dickens"' not in out
    assert 'string_value: "Jane Austen"' in out
    assert 'string_value: "Mark Twain"' not in out