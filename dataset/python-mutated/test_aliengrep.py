import pytest
from tests.fixtures import RunSemgrep

@pytest.mark.kinda_slow
@pytest.mark.parametrize('rule,target', [('rules/aliengrep/html.yaml', 'aliengrep/html.mustache'), ('rules/aliengrep/markdown.yaml', 'aliengrep/markdown.md'), ('rules/aliengrep/httpresponse.yaml', 'aliengrep/httpresponse.txt'), ('rules/aliengrep/dockerfile.yaml', 'aliengrep/dockerfile'), ('rules/aliengrep/multi-lines.yaml', 'aliengrep/multi-lines.java'), ('rules/aliengrep/terraform.yaml', 'aliengrep/terraform.tf'), ('rules/aliengrep/begin-end.yaml', 'aliengrep/begin-end.log'), ('rules/aliengrep/long-match.yaml', 'aliengrep/long-match.txt')])
def test_aliengrep(run_semgrep_in_tmp: RunSemgrep, snapshot, rule, target):
    if False:
        while True:
            i = 10
    snapshot.assert_match(run_semgrep_in_tmp(rule, target_name=target).stdout, 'results.json')

@pytest.mark.osemfail
@pytest.mark.kinda_slow
@pytest.mark.parametrize('rule,target', [('rules/aliengrep/nosem-html.yaml', 'aliengrep/nosem.html')])
def test_aliengrep_nosem(run_semgrep_in_tmp: RunSemgrep, snapshot, rule, target):
    if False:
        i = 10
        return i + 15
    snapshot.assert_match(run_semgrep_in_tmp(rule, target_name=target, options=['--no-rewrite-rule-ids']).stdout, 'results.json')