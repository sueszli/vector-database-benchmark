import io
from tempfile import NamedTemporaryFile
from textwrap import dedent
import pytest
from ruamel.yaml import YAML
from semgrep.config_resolver import Config
from semgrep.config_resolver import parse_config_string
from semgrep.config_resolver import validate_single_rule
from semgrep.constants import RULES_KEY
from semgrep.error import InvalidRuleSchemaError

@pytest.mark.quick
def test_parse_taint_rules():
    if False:
        i = 10
        return i + 15
    yaml_contents = dedent('\n        rules:\n          - id: stupid_equal\n            pattern: $X == $X\n            message: Dude, $X == $X is always true (Unless X is NAN ...)\n            languages: [python, javascript]\n            severity: WARNING\n          - id: stupid_equal2\n            mode: search\n            pattern: $X == $X\n            message: Dude, $X == $X is always true (Unless X is NAN ...)\n            languages: [python, javascript]\n            severity: WARNING\n          - id: example_id\n            mode: taint\n            pattern-sources:\n              - pattern: source(...)\n              - pattern: source1(...)\n            pattern-sinks:\n              - pattern: sink(...)\n              - pattern: sink1(...)\n              - pattern: eval(...)\n            pattern-sanitizers:\n              - pattern: sanitize(...)\n              - pattern: sanitize1(...)\n            message: A user input source() went into a dangerous sink()\n            languages: [python, javascript]\n            severity: WARNING\n        ')
    yaml = parse_config_string('testfile', yaml_contents, 'file.py')
    config = yaml['testfile'].value
    rules = config.get(RULES_KEY)
    for rule_dict in rules.value:
        validate_single_rule('testfile', rule_dict)
    assert True

@pytest.mark.quick
def test_multiple_configs():
    if False:
        while True:
            i = 10
    config1 = dedent('\n        rules:\n        - id: rule1\n          pattern: $X == $X\n          languages: [python]\n          severity: INFO\n          message: bad\n        ')
    config2 = dedent('\n        rules:\n        - id: rule2\n          pattern: $X == $Y\n          languages: [python]\n          severity: INFO\n          message: good\n        - id: rule3\n          pattern: $X < $Y\n          languages: [c]\n          severity: INFO\n          message: doog\n        ')
    with NamedTemporaryFile() as tf1, NamedTemporaryFile() as tf2:
        tf1.write(config1.encode('utf-8'))
        tf2.write(config2.encode('utf-8'))
        tf1.flush()
        tf2.flush()
        config_list = [tf1.name, tf2.name]
        (config, errors) = Config.from_config_list(config_list, None)
        assert not errors
        rules = config.get_rules(True)
        assert len(rules) == 3
        assert {'rule1', 'rule2', 'rule3'} == {rule.id for rule in rules}

@pytest.mark.quick
def test_default_yaml_type_safe():
    if False:
        return 10
    s = '!!python/object/apply:os.system ["echo Hello world"]'
    default_yaml = YAML()
    assert default_yaml.load(io.StringIO(s)) == ['echo Hello world']
    rt_yaml = YAML(typ='rt')
    assert rt_yaml.load(io.StringIO(s)) == ['echo Hello world']
    unsafe_yaml = YAML(typ='unsafe')
    assert unsafe_yaml.load(io.StringIO(s)) == 0

@pytest.mark.quick
def test_invalid_metavariable_regex():
    if False:
        print('Hello World!')
    rule = dedent('\n        rules:\n        - id: boto3-internal-network\n          patterns:\n          - pattern-inside: $MODULE.client(host=$HOST)\n          - metavariable-regex:\n              metavariable: $HOST\n              regex: \'192.168\\.\\d{1,3}\\.\\d{1,3}\'\n              metavariable: $MODULE\n              regex: (boto|boto3)\n          message: "Boto3 connection to internal network"\n          languages: [python]\n          severity: ERROR\n        ')
    with pytest.raises(InvalidRuleSchemaError):
        parse_config_string('testfile', rule, None)

@pytest.mark.quick
def test_invalid_metavariable_comparison():
    if False:
        print('Hello World!')
    rule = dedent('\n        rules:\n        - id: boto3-internal-network\n          patterns:\n          - pattern-inside: $MODULE.client(host=$HOST, port=$PORT)\n          - metavariable-comparison:\n              metavariable: $PORT\n              comparison: $PORT > 9999\n              metavariable: $MODULE\n              comparison: \'(server|servers)\'\n          message: "Boto3 connection to internal network"\n          languages: [python]\n          severity: ERROR\n        ')
    with pytest.raises(InvalidRuleSchemaError):
        parse_config_string('testfile', rule, None)

@pytest.mark.quick
def test_invalid_metavariable_comparison2():
    if False:
        return 10
    rule = dedent('\n        rules:\n        - id: boto3-internal-network\n          patterns:\n          - pattern-inside: $MODULE.client(host=$HOST, port=$PORT)\n          - metavariable-comparison:\n              metavariable: $PORT\n              comparison: $PORT > 9999\n              metavariable: $MODULE\n              regex: \'(server|servers)\'\n          message: "Boto3 connection to internal network"\n          languages: [python]\n          severity: ERROR\n        ')
    with pytest.raises(InvalidRuleSchemaError):
        parse_config_string('testfile', rule, None)

@pytest.mark.quick
def test_invalid_pattern_child():
    if False:
        i = 10
        return i + 15
    rule = dedent('\n        rules:\n        - id: blah\n          message: blah\n          severity: INFO\n          languages: [python]\n          patterns:\n          - pattern-either:\n            - pattern: $X == $Y\n            - pattern-not: $Z == $Z\n        ')
    with pytest.raises(InvalidRuleSchemaError):
        parse_config_string('testfile', rule, None)

@pytest.mark.quick
def test_invalid_rule_with_null():
    if False:
        for i in range(10):
            print('nop')
    rule = dedent('\n        rules:\n        - id: blah\n          message: ~\n          severity: INFO\n          languages: [python]\n          patterns:\n          - pattern-either:\n            - pattern: $X == $Y\n            - pattern-not: $Z == $Z\n        ')
    with pytest.raises(InvalidRuleSchemaError):
        parse_config_string('testfile', rule, None)