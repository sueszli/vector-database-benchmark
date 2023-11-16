import pytest
from tests.fixtures import RunSemgrep

@pytest.mark.kinda_slow
@pytest.mark.parametrize('rule,target', [('rules/join_rules/user-input-escaped-with-safe.yaml', 'join_rules/user-input-escaped-with-safe'), ('rules/join_rules/user-input-with-unescaped-extension.yaml', 'join_rules/user-input-with-unescaped-extension'), ('rules/join_rules/multiple-rules.yaml', 'join_rules/user-input-with-unescaped-extension'), ('rules/join_rules/inline/inline-rules.yaml', 'join_rules/user-input-with-unescaped-extension'), ('rules/join_rules/inline/taint.yaml', 'join_rules/user-input-with-unescaped-extension')])
@pytest.mark.osemfail
def test_join_rules(run_semgrep_in_tmp: RunSemgrep, snapshot, rule, target):
    if False:
        for i in range(10):
            print('nop')
    snapshot.assert_match(run_semgrep_in_tmp(rule, target_name=target).stdout, 'results.json')

@pytest.mark.kinda_slow
@pytest.mark.parametrize('rule,target', [('rules/join_rules/recursive/java-callgraph-example/vulnado-sqli.yaml', 'join_rules/recursive/java-callgraph-example/vulnado'), ('rules/join_rules/recursive/java-callgraph-example/vulnado-sqli.yaml', 'join_rules/recursive/java-callgraph-example/vulnado-chain-broken')])
@pytest.mark.osemfail
def test_recursive_join_rules(run_semgrep_in_tmp: RunSemgrep, snapshot, rule, target):
    if False:
        i = 10
        return i + 15
    snapshot.assert_match(run_semgrep_in_tmp(rule, target_name=target).stdout, 'results.json')