import google.auth
import pytest
import sync_api_with_long_timeout

def test_long_timeout(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        while True:
            i = 10
    request_file_name = 'resources/sync_request.json'
    (_, project_id) = google.auth.default()
    sync_api_with_long_timeout.long_timeout(request_file_name, project_id)
    (out, _) = capsys.readouterr()
    expected_strings = ['routes', 'visits', 'transitions', 'metrics']
    for expected_string in expected_strings:
        assert expected_string in out