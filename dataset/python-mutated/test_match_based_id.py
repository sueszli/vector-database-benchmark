import json
import shutil
from pathlib import Path
import pytest
from tests.fixtures import RunSemgrep
from semgrep.constants import OutputFormat

@pytest.mark.kinda_slow
def test_duplicate_matches_indexing(run_semgrep_in_tmp: RunSemgrep, snapshot):
    if False:
        while True:
            i = 10
    (results, _errors) = run_semgrep_in_tmp('rules/match_based_id/duplicates.yaml', target_name='match_based_id/duplicates', output_format=OutputFormat.JSON, clean_fingerprint=False)
    snapshot.assert_match(results, 'results.json')

@pytest.mark.kinda_slow
@pytest.mark.parametrize('rule,target_name,expect_change', [])
def test_id_change(run_semgrep_on_copied_files: RunSemgrep, tmp_path, rule, target_name, expect_change):
    if False:
        while True:
            i = 10
    '\n    Ensures that match-based IDs are resistant to various types of changes in code.\n\n    These changes are enumerated in\n       targets / match_based_id / (before|after) / <target_name>\n\n    To edit these cases, edit these files directly. To add new cases, add a corresponding pair\n    of files, and update the parameterization above.\n\n    :param rule: The Semgrep rule that should trigger a finding\n    :param target: The filename of the target pair\n    :param expect_change: Whether or not to expect an ID change\n    '
    static_target = tmp_path / 'targets' / ('_match_based_id' + Path(target_name).suffix)

    def run_on_target(subpath):
        if False:
            for i in range(10):
                print('nop')
        source_target = tmp_path / 'targets' / 'match_based_id' / subpath / target_name
        shutil.copy(source_target, static_target)
        (results, _) = run_semgrep_on_copied_files(rule, target_name=static_target, output_format=OutputFormat.JSON, clean_fingerprint=False)
        return json.loads(results)['results'][0]['extra']['fingerprint']
    before_id = run_on_target('before')
    after_id = run_on_target('after')
    assert (after_id != before_id) == expect_change

@pytest.mark.kinda_slow
@pytest.mark.parametrize('rule,target_name,expect_change', [('rules/match_based_id/formatting.yaml', 'formatting.c', False), ('rules/match_based_id/formatting.yaml', 'ellipse.c', False), ('rules/taint.yaml', 'taint.py', False), ('rules/match_based_id/operator.yaml', 'operator.c', True), ('rules/match_based_id/formatting.yaml', 'meta-change.c', True), ('rules/match_based_id/join.yaml', 'join.py', True)])
@pytest.mark.osemfail
def test_id_change_osemfail(run_semgrep_on_copied_files: RunSemgrep, tmp_path, rule, target_name, expect_change):
    if False:
        while True:
            i = 10
    test_id_change(run_semgrep_on_copied_files, tmp_path, rule, target_name, expect_change)