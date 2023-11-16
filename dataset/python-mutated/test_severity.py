import pytest
from tests.fixtures import RunSemgrep

@pytest.mark.kinda_slow
def test_severity_error(run_semgrep_in_tmp: RunSemgrep, snapshot):
    if False:
        print('Hello World!')
    json_str = run_semgrep_in_tmp('rules/inside.yaml', options=['--severity', 'ERROR']).stdout
    assert json_str != ''
    assert '"severity": "INFO"' not in json_str
    assert '"severity": "WARNING"' not in json_str
    snapshot.assert_match(json_str, 'results.json')

@pytest.mark.kinda_slow
def test_severity_info(run_semgrep_in_tmp: RunSemgrep, snapshot):
    if False:
        while True:
            i = 10
    snapshot.assert_match(run_semgrep_in_tmp('rules/inside.yaml', options=['--severity', 'INFO']).stdout, 'results.json')

@pytest.mark.kinda_slow
def test_severity_warning(run_semgrep_in_tmp: RunSemgrep, snapshot):
    if False:
        return 10
    snapshot.assert_match(run_semgrep_in_tmp('rules/inside.yaml', options=['--severity', 'WARNING']).stdout, 'results.json')

@pytest.mark.kinda_slow
def test_severity_multiple(run_semgrep_in_tmp: RunSemgrep, snapshot):
    if False:
        i = 10
        return i + 15
    snapshot.assert_match(run_semgrep_in_tmp('rules/inside.yaml', options=['--severity', 'INFO', '--severity', 'WARNING']).stdout, 'results.json')