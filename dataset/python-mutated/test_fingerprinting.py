import pytest
from sentry.grouping.api import get_default_grouping_config_dict
from sentry.grouping.fingerprinting import FingerprintingRules, InvalidFingerprintingConfig
from tests.sentry.grouping import with_fingerprint_input
GROUPING_CONFIG = get_default_grouping_config_dict()

def test_basic_parsing(insta_snapshot):
    if False:
        for i in range(10):
            print('nop')
    rules = FingerprintingRules.from_config_string('\n# This is a config\ntype:DatabaseUnavailable                        -> DatabaseUnavailable\nfunction:assertion_failed module:foo            -> AssertionFailed, foo\napp:true                                        -> aha\napp:true                                        -> {{ default }}\n!path:**/foo/**                                 -> everything\n!"path":**/foo/**                               -> everything\nlogger:sentry.*                                 -> logger-, {{ logger }}\nmessage:"\\x\\xff"                              -> stuff\nlogger:sentry.*                                 -> logger-{{ logger }}, title="Message from {{ logger }}"\nlogger:sentry.*                                 -> logger-{{ logger }} title="Message from {{ logger }}"\n')
    assert rules._to_config_structure() == {'rules': [{'matchers': [['type', 'DatabaseUnavailable']], 'fingerprint': ['DatabaseUnavailable'], 'attributes': {}}, {'matchers': [['function', 'assertion_failed'], ['module', 'foo']], 'fingerprint': ['AssertionFailed', 'foo'], 'attributes': {}}, {'matchers': [['app', 'true']], 'fingerprint': ['aha'], 'attributes': {}}, {'matchers': [['app', 'true']], 'fingerprint': ['{{ default }}'], 'attributes': {}}, {'matchers': [['!path', '**/foo/**']], 'fingerprint': ['everything'], 'attributes': {}}, {'matchers': [['!path', '**/foo/**']], 'fingerprint': ['everything'], 'attributes': {}}, {'matchers': [['logger', 'sentry.*']], 'fingerprint': ['logger-', '{{ logger }}'], 'attributes': {}}, {'matchers': [['message', '\\xÿ']], 'fingerprint': ['stuff'], 'attributes': {}}, {'matchers': [['logger', 'sentry.*']], 'fingerprint': ['logger-', '{{ logger }}'], 'attributes': {'title': 'Message from {{ logger }}'}}, {'matchers': [['logger', 'sentry.*']], 'fingerprint': ['logger-', '{{ logger }}'], 'attributes': {'title': 'Message from {{ logger }}'}}], 'version': 1}
    assert FingerprintingRules._from_config_structure(rules._to_config_structure())._to_config_structure() == rules._to_config_structure()

def test_rule_export():
    if False:
        return 10
    rules = FingerprintingRules.from_config_string('\nlogger:sentry.*                                 -> logger, {{ logger }}, title="Message from {{ logger }}"\n')
    assert rules.rules[0].to_json() == {'attributes': {'title': 'Message from {{ logger }}'}, 'fingerprint': ['logger', '{{ logger }}'], 'matchers': [['logger', 'sentry.*']]}

def test_parsing_errors():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(InvalidFingerprintingConfig):
        FingerprintingRules.from_config_string('invalid.message:foo -> bar')

def test_automatic_argument_splitting():
    if False:
        for i in range(10):
            print('nop')
    rules = FingerprintingRules.from_config_string('\nlogger:test -> logger-{{ logger }}\nlogger:test -> logger-, {{ logger }}\nlogger:test2 -> logger-{{ logger }}-{{ level }}\nlogger:test2 -> logger-, {{ logger }}, -, {{ level }}\n')
    assert rules._to_config_structure() == {'rules': [{'matchers': [['logger', 'test']], 'fingerprint': ['logger-', '{{ logger }}'], 'attributes': {}}, {'matchers': [['logger', 'test']], 'fingerprint': ['logger-', '{{ logger }}'], 'attributes': {}}, {'matchers': [['logger', 'test2']], 'fingerprint': ['logger-', '{{ logger }}', '-', '{{ level }}'], 'attributes': {}}, {'matchers': [['logger', 'test2']], 'fingerprint': ['logger-', '{{ logger }}', '-', '{{ level }}'], 'attributes': {}}], 'version': 1}

def test_discover_field_parsing(insta_snapshot):
    if False:
        for i in range(10):
            print('nop')
    rules = FingerprintingRules.from_config_string('\n# This is a config\nerror.type:DatabaseUnavailable                        -> DatabaseUnavailable\nstack.function:assertion_failed stack.module:foo      -> AssertionFailed, foo\napp:true                                        -> aha\napp:true                                        -> {{ default }}\n')
    assert rules._to_config_structure() == {'rules': [{'matchers': [['type', 'DatabaseUnavailable']], 'fingerprint': ['DatabaseUnavailable'], 'attributes': {}}, {'matchers': [['function', 'assertion_failed'], ['module', 'foo']], 'fingerprint': ['AssertionFailed', 'foo'], 'attributes': {}}, {'matchers': [['app', 'true']], 'fingerprint': ['aha'], 'attributes': {}}, {'matchers': [['app', 'true']], 'fingerprint': ['{{ default }}'], 'attributes': {}}], 'version': 1}
    assert FingerprintingRules._from_config_structure(rules._to_config_structure())._to_config_structure() == rules._to_config_structure()

@with_fingerprint_input('input')
def test_event_hash_variant(insta_snapshot, input):
    if False:
        for i in range(10):
            print('nop')
    (config, evt) = input.create_event()

    def dump_variant(v):
        if False:
            i = 10
            return i + 15
        rv = v.as_dict()
        for key in ('hash', 'description', 'config'):
            rv.pop(key, None)
        if 'component' in rv:
            for key in ('id', 'name', 'values'):
                rv['component'].pop(key, None)
        return rv
    insta_snapshot({'config': config.to_json(), 'fingerprint': evt.data['fingerprint'], 'title': evt.data['title'], 'variants': {k: dump_variant(v) for (k, v) in evt.get_grouping_variants(force_config=GROUPING_CONFIG).items()}})