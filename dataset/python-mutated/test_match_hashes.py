from pathlib import Path
from textwrap import dedent
import pytest
import semgrep.semgrep_interfaces.semgrep_output_v1 as out
from semgrep.config_resolver import parse_config_string
from semgrep.rule import Rule
from semgrep.rule_match import RuleMatch
from semgrep.rule_match import RuleMatchSet

@pytest.fixture
def eqeq_rule() -> Rule:
    if False:
        for i in range(10):
            print('nop')
    config = parse_config_string('testfile', dedent('\n        rules:\n        - id: rule_id\n          pattern: $X == $X\n          languages: [python]\n          severity: INFO\n          message: bad\n        '), None)
    return Rule.from_yamltree(config['testfile'].value['rules'].value[0])

@pytest.fixture
def double_eqeq_rule() -> Rule:
    if False:
        i = 10
        return i + 15
    config = parse_config_string('testfile', dedent('\n        rules:\n        - id: rule_id\n          pattern: |\n            $X == $X\n            $Y == $Y\n          languages: [python]\n          severity: INFO\n          message: bad\n        '), None)
    return Rule.from_yamltree(config['testfile'].value['rules'].value[0])

@pytest.fixture
def foo_contents() -> str:
    if False:
        for i in range(10):
            print('nop')
    return dedent('\n        # first line\n        def foo():\n            5 == 5 # nosem\n            5 == 5 # nosem\n            6 == 6 # nosem\n            5 == 5 # nosem\n        ').lstrip()

def get_rule_match(filepath='foo.py', start_line=3, end_line=3, rule_id='rule_id', metavars=None, metadata=None) -> RuleMatch:
    if False:
        while True:
            i = 10
    return RuleMatch(message='message', severity=out.MatchSeverity(out.Error()), match=out.CoreMatch(check_id=out.RuleId(rule_id), path=out.Fpath(filepath), start=out.Position(start_line, 0, start_line * 5), end=out.Position(end_line, 5, end_line * 5 + 5), extra=out.CoreMatchExtra(metavars=out.Metavars(metavars if metavars else {}), engine_kind=out.EngineKind(out.OSS()))), extra={'metavars': metavars if metavars else {}}, metadata=metadata if metadata else {})

@pytest.mark.quick
def test_code_hash_independent_of_filepath(mocker, foo_contents):
    if False:
        return 10
    mocker.patch.object(Path, 'open', mocker.mock_open(read_data=foo_contents))
    match_1 = get_rule_match(filepath='foo.py')
    match_2 = get_rule_match(filepath='bar/foo.py')
    assert match_1.syntactic_id != match_2.syntactic_id
    assert match_1.match_based_id != match_2.match_based_id
    assert match_1.code_hash == match_2.code_hash
    assert match_1.pattern_hash == match_2.pattern_hash

@pytest.mark.quick
def test_code_hash_independent_of_rulename(mocker, foo_contents):
    if False:
        print('Hello World!')
    mocker.patch.object(Path, 'open', mocker.mock_open(read_data=foo_contents))
    match_1 = get_rule_match(rule_id='first.rule.id')
    match_2 = get_rule_match(rule_id='second.rule.id')
    assert match_1.syntactic_id != match_2.syntactic_id
    assert match_1.match_based_id != match_2.match_based_id
    assert match_1.code_hash == match_2.code_hash
    assert match_1.pattern_hash == match_2.pattern_hash

@pytest.mark.quick
def test_code_hash_independent_of_index(mocker, eqeq_rule, foo_contents):
    if False:
        i = 10
        return i + 15
    mocker.patch.object(Path, 'open', mocker.mock_open(read_data=foo_contents))
    match_1 = get_rule_match(start_line=3, end_line=3)
    match_2 = get_rule_match(start_line=4, end_line=4)
    matches = RuleMatchSet(eqeq_rule)
    matches.update([match_1, match_2])
    matches = list(sorted(matches))
    assert matches[0].index == 0
    assert matches[1].index == 1
    assert matches[0].syntactic_id != matches[1].syntactic_id
    assert matches[0].match_based_id != matches[1].match_based_id
    assert matches[0].code_hash == matches[1].code_hash
    assert matches[0].pattern_hash == matches[1].pattern_hash

@pytest.mark.quick
def test_code_hash_changes_with_code(mocker, eqeq_rule, foo_contents):
    if False:
        for i in range(10):
            print('nop')
    mocker.patch.object(Path, 'open', mocker.mock_open(read_data=foo_contents))
    match_1 = get_rule_match(start_line=3, end_line=3, metavars={'$X': {'abstract_content': '5'}})
    match_2 = get_rule_match(start_line=5, end_line=5, metavars={'$X': {'abstract_content': '6'}})
    matches = RuleMatchSet(eqeq_rule)
    matches.update([match_1, match_2])
    matches = list(sorted(matches))
    assert matches[0].index == 0
    assert matches[1].index == 0
    assert matches[0].code_hash != matches[1].code_hash
    assert matches[0].pattern_hash != matches[1].pattern_hash

@pytest.mark.quick
def test_line_hashes_hash_correct_line(mocker, double_eqeq_rule, foo_contents):
    if False:
        while True:
            i = 10
    mocker.patch.object(Path, 'open', mocker.mock_open(read_data=foo_contents))
    match_1 = get_rule_match(start_line=4, end_line=5)
    match_2 = get_rule_match(start_line=5, end_line=6)
    matches = RuleMatchSet(double_eqeq_rule)
    matches.update([match_1, match_2])
    matches = list(sorted(matches))
    assert matches[0].start_line_hash != matches[0].end_line_hash
    assert matches[1].start_line_hash != matches[1].end_line_hash
    assert matches[0].start_line_hash == matches[1].end_line_hash
    assert matches[0].end_line_hash == matches[1].start_line_hash

@pytest.mark.quick
def test_same_code_hash_for_previous_scan_finding(mocker, foo_contents):
    if False:
        return 10
    "\n    For the reliable fixed status work, we start sending rules run during the previous\n    scan too.\n\n    As the engine can't process two rules with same rule.id, we override the rule.id for\n    previous scan findings with something unique. However, we store the original rule.id\n    in the metadata.\n\n    Before computing the match_based_id, we fetch the rule.id from the metadata and use\n    it to compute the match_based_id.\n\n    This test ensures that the match_based_id for the previous scan finding is same as\n    the match_based_id for the current scan finding.\n    "
    mocker.patch.object(Path, 'open', mocker.mock_open(read_data=foo_contents))
    curr_scan_metadata = {'semgrep.dev': {'rule': {'rule_id': 'rule_id', 'version_id': 'version1', 'url': 'https://semgrep.dev/r/python.eqeq-five', 'shortlink': 'https://sg.run/abcd'}, 'src': 'unchanged'}}
    prev_scan_metadata = {'semgrep.dev': {'rule': {'rule_id': 'rule_idversion1', 'version_id': 'version1', 'url': 'https://semgrep.dev/r/python.eqeq-five', 'shortlink': 'https://sg.run/abcd', 'rule_name': 'rule_id'}, 'src': 'previous-scan'}}
    curr_scan_match = get_rule_match(start_line=3, end_line=3, metadata=curr_scan_metadata, rule_id='rule_id')
    prev_scan_match = get_rule_match(start_line=3, end_line=3, metadata=prev_scan_metadata, rule_id='rule_idversion1')
    assert curr_scan_match.syntactic_id == prev_scan_match.syntactic_id
    assert curr_scan_match.match_based_id == prev_scan_match.match_based_id
    assert curr_scan_match.code_hash == prev_scan_match.code_hash
    assert prev_scan_match.pattern_hash == prev_scan_match.pattern_hash