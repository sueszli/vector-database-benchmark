"""
Tests for parse rates from semgrep.metrics.

Ensures that the parse errors reported from core are correctly picked up by the
CLI.
"""
import json
import sys
from pathlib import Path
from shutil import copytree
import pytest
from tests.conftest import TESTS_PATH
from tests.semgrep_runner import SemgrepRunner
from semgrep.cli import cli

@pytest.mark.quick
@pytest.mark.skipif(sys.version_info < (3, 8), reason="snapshotting mock call kwargs doesn't work on py3.7")
@pytest.mark.osemfail
def test_parse_metrics(tmp_path, snapshot, mocker, monkeypatch):
    if False:
        i = 10
        return i + 15
    mock_post = mocker.patch('requests.post')
    copytree(Path(TESTS_PATH / 'e2e' / 'targets' / 'parse_metrics').resolve(), tmp_path / 'parse_metrics')
    monkeypatch.chdir(tmp_path / 'parse_metrics')
    SemgrepRunner(use_click_runner=True).invoke(cli, ['scan', '--config=rules.yaml', '--metrics=on'])
    payload = json.loads(mock_post.call_args.kwargs['data'])
    snapshot.assert_match(json.dumps(payload['parse_rate'], indent=2, sort_keys=True), 'parse-rates.json')