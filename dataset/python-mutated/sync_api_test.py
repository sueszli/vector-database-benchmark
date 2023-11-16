import google.auth
import pytest
import sync_api

def test_call_sync_api(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        i = 10
        return i + 15
    (_, project_id) = google.auth.default()
    sync_api.call_sync_api(project_id)
    (out, _) = capsys.readouterr()
    expected_strings = ['routes', 'visits', 'transitions', 'metrics']
    for expected_string in expected_strings:
        assert expected_string in out