import os
import inspect_column_values_w_custom_hotwords as inspect_content
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_inspect_column_values_w_custom_hotwords(capsys: pytest.CaptureFixture) -> None:
    if False:
        i = 10
        return i + 15
    table_data = {'header': ['Fake Social Security Number', 'Real Social Security Number'], 'rows': [['111-11-1111', '222-22-2222'], ['987-23-1234', '333-33-3333'], ['678-12-0909', '444-44-4444']]}
    inspect_content.inspect_column_values_w_custom_hotwords(GCLOUD_PROJECT, table_data['header'], table_data['rows'], ['US_SOCIAL_SECURITY_NUMBER'], 'Fake Social Security Number')
    (out, _) = capsys.readouterr()
    assert 'Info type: US_SOCIAL_SECURITY_NUMBER' in out
    assert '222-22-2222' in out
    assert '111-11-1111' not in out