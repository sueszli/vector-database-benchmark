import pytest
from tests.fixtures import RunSemgrep

@pytest.mark.kinda_slow
@pytest.mark.osemfail
def test_metavariable_regex_const_prop(run_semgrep_in_tmp: RunSemgrep, snapshot):
    if False:
        while True:
            i = 10
    snapshot.assert_match(run_semgrep_in_tmp('rules/metavariable-regex-const-prop.yaml', target_name='metavariable_propagation/metavariable-regex-const-prop.dockerfile').stdout, 'results.json')