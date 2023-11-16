import pytest
from tests.fixtures import RunSemgrep

@pytest.mark.kinda_slow
@pytest.mark.osemfail
def test_pro_rule_skipping(run_semgrep_in_tmp: RunSemgrep, snapshot):
    if False:
        return 10
    snapshot.assert_match(run_semgrep_in_tmp('rules/pro-rule-skipping.yaml', target_name='pro-rule-skipping/x.cls').stdout, 'results.json')

@pytest.mark.kinda_slow
@pytest.mark.osemfail
def test_pro_rule_skipping_no_parsing(run_semgrep_in_tmp: RunSemgrep, snapshot):
    if False:
        return 10
    snapshot.assert_match(run_semgrep_in_tmp('rules/pro-rule-skipping-no-parsing.yaml', target_name='pro-rule-skipping-no-parsing/x.cls').stdout, 'results.json')