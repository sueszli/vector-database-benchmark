"""
https://github.com/behave/behave/issues/1061
"""
from __future__ import absolute_import, print_function
from behave.parser import parse_feature
from tests.functional.test_tag_inheritance import get_inherited_tags, assert_inherited_tags_equal_to

class TestIssue(object):
    """Verifies that issue is fixed.
    Verifies basics that tag-inheritance mechanism works.

    .. seealso:: tests/functional/test_tag_inheritance.py
    """

    def test_scenario_inherits_tags_with_feature(self):
        if False:
            return 10
        'Verifies that issue #1047 is fixed.'
        text = u'\n            @feature_tag1\n            Feature: F1\n\n              @scenario_tag1\n              Scenario: S1\n            '
        this_feature = parse_feature(text)
        this_scenario = this_feature.scenarios[0]
        expected_tags = set(['scenario_tag1', 'feature_tag1'])
        assert this_scenario.effective_tags == expected_tags
        assert_inherited_tags_equal_to(this_scenario, ['feature_tag1'])

    def test_scenario_inherits_tags_with_rule(self):
        if False:
            i = 10
            return i + 15
        text = u'\n            @feature_tag1\n            Feature: F1\n              @rule_tag1 @rule_tag2\n              Rule: R1\n                @scenario_tag1\n                Scenario: S1\n            '
        this_feature = parse_feature(text)
        this_scenario = this_feature.rules[0].scenarios[0]
        inherited_tags = ['feature_tag1', 'rule_tag1', 'rule_tag2']
        expected_tags = set(['scenario_tag1']).union(inherited_tags)
        assert this_scenario.effective_tags == expected_tags
        assert_inherited_tags_equal_to(this_scenario, inherited_tags)

    def test_issue_scenario_inherits_tags_with_scenario_outline_and_rule(self):
        if False:
            print('Hello World!')
        text = u'\n            @feature_tag1\n            Feature: F1\n              @rule_tag1 @rule_tag2\n              Rule: R1\n                @scenario_tag1\n                Scenario Outline: S1\n                  Examples:\n                    | name  |\n                    | Alice |\n            '
        this_feature = parse_feature(text)
        this_scenario_outline = this_feature.rules[0].scenarios[0]
        this_scenario = this_scenario_outline.scenarios[0]
        inherited_tags = ['feature_tag1', 'rule_tag1', 'rule_tag2']
        expected_tags = set(['scenario_tag1']).union(inherited_tags)
        assert this_scenario.effective_tags == expected_tags
        assert_inherited_tags_equal_to(this_scenario, inherited_tags)