"""
Test the tag inheritance mechanism between model entities:

* Feature(s)
* Rule(s)
* ScenarioOutline(s)
* Scenario(s)

Tag inheritance mechanism:

* Inner model element inherits tags from its outer/parent elements
* Parametrized tags from a ScenarioOutline/ScenarioTemplate are filtered out

EXAMPLES:

* Scenario inherits the tags of its Feature
* Scenario inherits the tags of its Rule
* Scenario derives its tags of its ScenarioOutline (and Examples table)

* Rule inherits tags of its Feature
* ScenarioOutline/ScenarioTemplate inherits tags from its Feature
* ScenarioOutline/ScenarioTemplate inherits tags from its Rule
"""
from __future__ import absolute_import, print_function
from behave.parser import parse_feature
import pytest

def get_inherited_tags(model_element):
    if False:
        print('Hello World!')
    inherited_tags = set(model_element.effective_tags).difference(model_element.tags)
    return sorted(inherited_tags)

def assert_tags_same_as_effective_tags(model_element):
    if False:
        while True:
            i = 10
    assert set(model_element.tags) == set(model_element.effective_tags)

def assert_inherited_tags_equal_to(model_element, expected_tags):
    if False:
        print('Hello World!')
    inherited_tags = get_inherited_tags(model_element)
    assert inherited_tags == expected_tags

def assert_no_tags_are_inherited(model_element):
    if False:
        while True:
            i = 10
    assert_inherited_tags_equal_to(model_element, [])

class TestTagInheritance4Feature(object):
    """A Feature is the outermost model element.
    Therefore, it cannot inherit any features.
    """

    @pytest.mark.parametrize('tags, case', [([], 'without tags'), (['feature_tag1', 'feature_tag2'], 'with tags')])
    def test_no_inherited_tags(self, tags, case):
        if False:
            for i in range(10):
                print('nop')
        tag_line = ' '.join(('@%s' % tag for tag in tags))
        text = u'\n            {tag_line}\n            Feature: F1\n            '.format(tag_line=tag_line)
        this_feature = parse_feature(text)
        assert this_feature.tags == tags
        assert this_feature.effective_tags == set(tags)
        assert_no_tags_are_inherited(this_feature)

class TestTagInheritance4Rule(object):

    def test_no_inherited_tags__without_feature_tags(self):
        if False:
            i = 10
            return i + 15
        text = u'\n        Feature: F1\n          @rule_tag1\n          Rule: R1\n        '
        this_feature = parse_feature(text)
        this_rule = this_feature.rules[0]
        assert this_feature.tags == []
        assert this_rule.tags == ['rule_tag1']
        assert_tags_same_as_effective_tags(this_rule)
        assert_no_tags_are_inherited(this_rule)

    def test_inherited_tags__with_feature_tags(self):
        if False:
            for i in range(10):
                print('nop')
        text = u'\n        @feature_tag1 @feature_tag2\n        Feature: F2\n          @rule_tag1\n          Rule: R2\n        '
        this_feature = parse_feature(text)
        this_rule = this_feature.rules[0]
        expected_feature_tags = ['feature_tag1', 'feature_tag2']
        assert this_feature.tags == expected_feature_tags
        assert this_rule.tags == ['rule_tag1']
        assert_inherited_tags_equal_to(this_rule, expected_feature_tags)

    def test_duplicated_tags_are_removed_from_inherited_tags(self):
        if False:
            print('Hello World!')
        text = u'\n        @feature_tag1 @duplicated_tag\n        Feature: F2\n          @rule_tag1 @duplicated_tag\n          Rule: R2\n        '
        this_feature = parse_feature(text)
        this_rule = this_feature.rules[0]
        assert this_feature.tags == ['feature_tag1', 'duplicated_tag']
        assert this_rule.tags == ['rule_tag1', 'duplicated_tag']
        assert_inherited_tags_equal_to(this_rule, ['feature_tag1'])

class TestTagInheritance4ScenarioOutline(object):

    def test_no_inherited_tags__without_feature_tags(self):
        if False:
            while True:
                i = 10
        text = u'\n        Feature: F3\n            @outline_tag1\n            Scenario Outline: T1\n        '
        this_feature = parse_feature(text)
        this_scenario_outline = this_feature.run_items[0]
        assert this_feature.tags == []
        assert this_scenario_outline.tags == ['outline_tag1']
        assert_no_tags_are_inherited(this_scenario_outline)

    def test_no_inherited_tags__without_feature_and_rule_tags(self):
        if False:
            return 10
        text = u'\n        Feature: F3\n          Rule: R3\n            @outline_tag1\n            Scenario Outline: T1\n        '
        this_feature = parse_feature(text)
        this_rule = this_feature.rules[0]
        this_scenario_outline = this_rule.run_items[0]
        assert this_feature.tags == []
        assert this_rule.tags == []
        assert this_scenario_outline.tags == ['outline_tag1']
        assert_no_tags_are_inherited(this_scenario_outline)

    def test_inherited_tags__with_feature_tags(self):
        if False:
            return 10
        text = u'\n        @feature_tag1 @feature_tag2\n        Feature: F3\n            @outline_tag1\n            Scenario Outline: T3\n        '
        this_feature = parse_feature(text)
        this_scenario_outline = this_feature.run_items[0]
        expected_feature_tags = ['feature_tag1', 'feature_tag2']
        assert this_feature.tags == expected_feature_tags
        assert this_scenario_outline.tags == ['outline_tag1']
        assert_inherited_tags_equal_to(this_scenario_outline, expected_feature_tags)

    def test_inherited_tags__with_rule_tags(self):
        if False:
            return 10
        text = u'\n        Feature: F3\n          @rule_tag1 @rule_tag2\n          Rule: R3\n            @outline_tag1\n            Scenario Outline: T3\n        '
        this_feature = parse_feature(text)
        this_rule = this_feature.rules[0]
        this_scenario_outline = this_rule.run_items[0]
        expected_rule_tags = ['rule_tag1', 'rule_tag2']
        assert this_feature.tags == []
        assert this_rule.tags == expected_rule_tags
        assert this_scenario_outline.tags == ['outline_tag1']
        assert_inherited_tags_equal_to(this_scenario_outline, expected_rule_tags)

    def test_inherited_tags__with_feature_and_rule_tags(self):
        if False:
            i = 10
            return i + 15
        text = u'\n        @feature_tag1\n        Feature: F3\n          @rule_tag1 @rule_tag2\n          Rule: R3\n            @outline_tag1\n            Scenario Outline: T3\n        '
        this_feature = parse_feature(text)
        this_rule = this_feature.rules[0]
        this_scenario_outline = this_rule.run_items[0]
        expected_feature_tags = ['feature_tag1']
        expected_rule_tags = ['rule_tag1', 'rule_tag2']
        expected_inherited_tags = ['feature_tag1', 'rule_tag1', 'rule_tag2']
        assert this_feature.tags == expected_feature_tags
        assert this_rule.tags == expected_rule_tags
        assert this_scenario_outline.tags == ['outline_tag1']
        assert_inherited_tags_equal_to(this_scenario_outline, expected_inherited_tags)

    def test_duplicated_tags_are_removed_from_inherited_tags(self):
        if False:
            for i in range(10):
                print('nop')
        text = u'\n        @feature_tag1 @duplicated_tag\n        Feature: F3\n          @rule_tag1 @duplicated_tag\n          Rule: R3\n            @outline_tag1 @duplicated_tag\n            Scenario Outline: T3\n        '
        this_feature = parse_feature(text)
        this_rule = this_feature.rules[0]
        this_scenario_outline = this_rule.run_items[0]
        assert this_feature.tags == ['feature_tag1', 'duplicated_tag']
        assert this_rule.tags == ['rule_tag1', 'duplicated_tag']
        assert this_scenario_outline.tags == ['outline_tag1', 'duplicated_tag']
        assert_inherited_tags_equal_to(this_scenario_outline, ['feature_tag1', 'rule_tag1'])

class TestTagInheritance4Scenario(object):

    def test_no_inherited_tags__without_feature_tags(self):
        if False:
            print('Hello World!')
        text = u'\n        Feature: F4\n            @scenario_tag1\n            Scenario: S4\n        '
        this_feature = parse_feature(text)
        this_scenario = this_feature.scenarios[0]
        assert this_feature.tags == []
        assert this_scenario.tags == ['scenario_tag1']
        assert_no_tags_are_inherited(this_scenario)

    def test_no_inherited_tags__without_feature_and_rule_tags(self):
        if False:
            i = 10
            return i + 15
        text = u'\n        Feature: F4\n          Rule: R4\n            @scenario_tag1\n            Scenario: S4\n        '
        this_feature = parse_feature(text)
        this_rule = this_feature.rules[0]
        this_scenario = this_rule.scenarios[0]
        assert this_feature.tags == []
        assert this_rule.tags == []
        assert this_scenario.tags == ['scenario_tag1']
        assert_no_tags_are_inherited(this_scenario)

    def test_inherited_tags__with_feature_tags(self):
        if False:
            return 10
        text = u'\n        @feature_tag1 @feature_tag2\n        Feature: F4\n            @scenario_tag1\n            Scenario: S4\n        '
        this_feature = parse_feature(text)
        this_scenario = this_feature.scenarios[0]
        expected_feature_tags = ['feature_tag1', 'feature_tag2']
        assert this_feature.tags == expected_feature_tags
        assert this_scenario.tags == ['scenario_tag1']
        assert_inherited_tags_equal_to(this_scenario, expected_feature_tags)

    def test_inherited_tags__with_rule_tags(self):
        if False:
            return 10
        text = u'\n        Feature: F3\n          @rule_tag1 @rule_tag2\n          Rule: R3\n            @scenario_tag1\n            Scenario: S4\n        '
        this_feature = parse_feature(text)
        this_rule = this_feature.rules[0]
        this_scenario = this_rule.scenarios[0]
        expected_rule_tags = ['rule_tag1', 'rule_tag2']
        assert this_feature.tags == []
        assert this_rule.tags == expected_rule_tags
        assert this_scenario.tags == ['scenario_tag1']
        assert_inherited_tags_equal_to(this_scenario, expected_rule_tags)

    def test_inherited_tags__with_feature_and_rule_tags(self):
        if False:
            i = 10
            return i + 15
        text = u'\n        @feature_tag1\n        Feature: F4\n          @rule_tag1 @rule_tag2\n          Rule: R4\n            @scenario_tag1\n            Scenario: S4\n        '
        this_feature = parse_feature(text)
        this_rule = this_feature.rules[0]
        this_scenario = this_rule.scenarios[0]
        expected_feature_tags = ['feature_tag1']
        expected_rule_tags = ['rule_tag1', 'rule_tag2']
        expected_inherited_tags = ['feature_tag1', 'rule_tag1', 'rule_tag2']
        assert this_feature.tags == expected_feature_tags
        assert this_rule.tags == expected_rule_tags
        assert this_scenario.tags == ['scenario_tag1']
        assert_inherited_tags_equal_to(this_scenario, expected_inherited_tags)

    def test_duplicated_tags_are_removed_from_inherited_tags(self):
        if False:
            for i in range(10):
                print('nop')
        text = u'\n        @feature_tag1 @duplicated_tag\n        Feature: F4\n          @rule_tag1 @duplicated_tag\n          Rule: R4\n            @scenario_tag1 @duplicated_tag\n            Scenario: S4\n        '
        this_feature = parse_feature(text)
        this_rule = this_feature.rules[0]
        this_scenario = this_rule.scenarios[0]
        assert this_feature.tags == ['feature_tag1', 'duplicated_tag']
        assert this_rule.tags == ['rule_tag1', 'duplicated_tag']
        assert this_scenario.tags == ['scenario_tag1', 'duplicated_tag']
        assert_inherited_tags_equal_to(this_scenario, ['feature_tag1', 'rule_tag1'])

class TestTagInheritance4ScenarioFromTemplate(object):
    """Test tag inheritance for scenarios from a ScenarioOutline or
    ScenarioTemplate (as alias for ScenarioOutline).

    SCENARIO TEMPLATE MECHANISM::

        scenario_template := scenario_outline
        scenario.tags := scenario_template.tags + scenario_template.examples[i].tags
    """

    def test_no_inherited_tags__without_feature_tags(self):
        if False:
            print('Hello World!')
        text = u'\n        Feature: F5\n            @template_tag1\n            Scenario Outline: T5\n              Given I meet "<name>"\n\n              @examples_tag1\n              Examples:\n                | name |\n                | Alice |\n        '
        this_feature = parse_feature(text)
        this_scenario_outline = this_feature.run_items[0]
        this_scenario = this_scenario_outline.scenarios[0]
        assert this_feature.tags == []
        assert this_scenario_outline.tags == ['template_tag1']
        assert this_scenario.tags == ['template_tag1', 'examples_tag1']
        assert_no_tags_are_inherited(this_scenario)

    def test_no_inherited_tags__without_feature_and_rule_tags(self):
        if False:
            while True:
                i = 10
        text = u'\n        Feature: F5\n          Rule: R5\n            @template_tag1\n            Scenario Outline: T5\n              Given I meet "<name>"\n\n              @examples_tag1\n              Examples:\n                | name |\n                | Alice |\n        '
        this_feature = parse_feature(text)
        this_rule = this_feature.rules[0]
        this_scenario_outline = this_rule.run_items[0]
        this_scenario = this_scenario_outline.scenarios[0]
        assert this_feature.tags == []
        assert this_rule.tags == []
        assert this_scenario_outline.tags == ['template_tag1']
        assert this_scenario.tags == ['template_tag1', 'examples_tag1']
        assert_no_tags_are_inherited(this_scenario)

    def test_inherited_tags__with_feature_tags(self):
        if False:
            i = 10
            return i + 15
        text = u'\n        @feature_tag1 @feature_tag2\n        Feature: F5\n            @template_tag1\n            Scenario Outline: T5\n              Given I meet "<name>"\n\n              Examples:\n                | name |\n                | Alice |\n        '
        this_feature = parse_feature(text)
        this_scenario_outline = this_feature.run_items[0]
        this_scenario = this_scenario_outline.scenarios[0]
        expected_feature_tags = ['feature_tag1', 'feature_tag2']
        assert this_feature.tags == expected_feature_tags
        assert this_scenario.tags == ['template_tag1']
        assert_inherited_tags_equal_to(this_scenario, expected_feature_tags)

    def test_inherited_tags__with_rule_tags(self):
        if False:
            i = 10
            return i + 15
        text = u'\n        Feature: F5\n          @rule_tag1 @rule_tag2\n          Rule: R5\n\n            @template_tag1\n            Scenario Outline: T5\n              Given I meet "<name>"\n\n              Examples:\n                | name |\n                | Alice |\n        '
        this_feature = parse_feature(text)
        this_rule = this_feature.rules[0]
        this_scenario_outline = this_feature.run_items[0]
        this_scenario = this_scenario_outline.scenarios[0]
        expected_rule_tags = ['rule_tag1', 'rule_tag2']
        assert this_feature.tags == []
        assert this_rule.tags == expected_rule_tags
        assert this_scenario.tags == ['template_tag1']
        assert_inherited_tags_equal_to(this_scenario, expected_rule_tags)

    def test_inherited_tags__with_feature_and_rule_tags(self):
        if False:
            i = 10
            return i + 15
        text = u'\n        @feature_tag1\n        Feature: F4\n          @rule_tag1 @rule_tag2\n          Rule: R4\n\n            @template_tag1\n            Scenario Outline: T5\n              Given I meet "<name>"\n\n              Examples:\n                | name |\n                | Alice |\n        '
        this_feature = parse_feature(text)
        this_rule = this_feature.rules[0]
        this_scenario_outline = this_rule.run_items[0]
        this_scenario = this_scenario_outline.scenarios[0]
        expected_feature_tags = ['feature_tag1']
        expected_rule_tags = ['rule_tag1', 'rule_tag2']
        expected_inherited_tags = expected_feature_tags + expected_rule_tags
        assert this_feature.tags == expected_feature_tags
        assert this_rule.tags == expected_rule_tags
        assert this_scenario.tags == ['template_tag1']
        assert_inherited_tags_equal_to(this_scenario, expected_inherited_tags)

    def test_tags_are_derived_from_template(self):
        if False:
            for i in range(10):
                print('nop')
        text = u'\n        Feature: F5\n\n            @template_tag1 @param_name_<name>\n            Scenario Outline: T5\n              Given I meet "<name>"\n\n              Examples:\n                | name |\n                | Alice |\n        '
        this_feature = parse_feature(text)
        this_scenario_template = this_feature.run_items[0]
        this_scenario = this_scenario_template.scenarios[0]
        assert this_feature.tags == []
        assert this_scenario_template.tags == ['template_tag1', 'param_name_<name>']
        assert this_scenario.tags == ['template_tag1', 'param_name_Alice']
        assert_no_tags_are_inherited(this_scenario)

    def test_tags_are_derived_from_template_examples_for_table_row(self):
        if False:
            while True:
                i = 10
        text = u'\n        Feature: F5\n          Rule: R5\n            Scenario Outline: T5\n              Given I meet "<name>"\n\n              @examples_tag1\n              Examples:\n                | name |\n                | Alice |\n        '
        this_feature = parse_feature(text)
        this_rule = this_feature.rules[0]
        this_scenario_outline = this_rule.run_items[0]
        this_scenario = this_scenario_outline.scenarios[0]
        assert this_feature.tags == []
        assert this_scenario.tags == ['examples_tag1']
        assert_no_tags_are_inherited(this_scenario)

    def test_duplicated_tags_are_removed_from_inherited_tags(self):
        if False:
            i = 10
            return i + 15
        text = u'\n        @feature_tag1 @duplicated_tag\n        Feature: F4\n          @rule_tag1 @duplicated_tag\n          Rule: R4\n\n            @template_tag1 @duplicated_tag\n            Scenario Outline: T5\n              Given I meet "<name>"\n\n              @examples_tag1\n              Examples:\n                | name |\n                | Alice |\n        '
        this_feature = parse_feature(text)
        this_rule = this_feature.rules[0]
        this_scenario_template = this_rule.scenarios[0]
        this_scenario = this_scenario_template.scenarios[0]
        assert this_feature.tags == ['feature_tag1', 'duplicated_tag']
        assert this_rule.tags == ['rule_tag1', 'duplicated_tag']
        assert this_scenario.tags == ['template_tag1', 'duplicated_tag', 'examples_tag1']
        assert_inherited_tags_equal_to(this_scenario, ['feature_tag1', 'rule_tag1'])