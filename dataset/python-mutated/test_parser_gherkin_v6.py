"""
Tests for Gherkin parser with `Gherkin v6 grammar`_.

Gherkin v6 grammar extensions:

* Rule keyword added
* Aliases for Scenario, ScenarioOutline to better correspond to `Example Mapping`_

A Rule (or: business rule) allows to group multiple Scenarios::

    # -- RULE GRAMMAR PSEUDO-CODE:
    @tag1 @tag2
    Rule: Optional Rule Title...
        Description?        #< CARDINALITY: 0..1 (optional)
        Background?         #< CARDINALITY: 0..1 (optional)
        Scenario*           #< CARDINALITY: 0..N (many)
        ScenarioOutline*    #< CARDINALITY: 0..N (many)

Keyword aliases:

    | Concept          | Gherkin v6         | Alias (Gherkin v5) |
    | Scenario         | Example            | Scenario           |
    | Scenario Outline | Scenario Template  | Scenario Outline   |

.. seealso::

    * :class:~behave.parser:Parser`
    * `Gherkin v6 grammar`_

    EXAMPLE MAPPING:

    * Cucumber: `Example Mapping`_
    * Cucumber: `Example Mapping Webinar`_
    * https://docs.cucumber.io/bdd/example-mapping/
    * https://www.agilealliance.org/resources/sessions/example-mapping/

.. _`Gherkin v6 grammar`: https://github.com/cucumber/cucumber/blob/master/gherkin/gherkin.berp
.. _`Example Mapping`: https://cucumber.io/blog/example-mapping-introduction/
.. _`Example Mapping Webinar`: https://cucumber.io/blog/example-mapping-webinar/
"""
from __future__ import absolute_import, print_function
from behave.parser import parse_feature, parse_rule, Parser, ParserError
from behave.model import Feature, Rule, Scenario, ScenarioOutline, Background
import pytest

def assert_compare_steps(steps, expected):
    if False:
        for i in range(10):
            print('nop')
    have = [(s.step_type, s.keyword.strip(), s.name, s.text, s.table) for s in steps]
    assert have == expected

class TestGherkin6Parser(object):
    pass

class TestParser4Rule(object):

    @pytest.mark.smoke
    def test_parses_rule(self):
        if False:
            return 10
        text = u'\nFeature: With Rule\n\n  A feature description line 1.\n  A feature description line 2.\n\n  Rule: R1\n    Background: R1.Background_1\n      Given background step 1\n      When background step 2\n\n    Scenario: R1.Scenario_1\n      Given scenario step 1\n      When scenario step 2\n      Then scenario step 3\n'.lstrip()
        feature = parse_feature(text)
        rule1 = feature.rules[0]
        rule1_scenario1 = rule1.scenarios[0]
        assert feature.name == 'With Rule'
        assert feature.description == ['A feature description line 1.', 'A feature description line 2.']
        assert feature.background is None
        assert len(feature.rules) == 1
        assert len(feature.scenarios) == 0
        assert rule1.name == 'R1'
        assert rule1.tags == []
        assert rule1.parent is feature
        assert rule1.feature is feature
        assert rule1.background is not None
        assert_compare_steps(rule1.background.steps, [('given', 'Given', 'background step 1', None, None), ('when', 'When', 'background step 2', None, None)])
        assert len(rule1.scenarios) == 1
        assert rule1_scenario1.background is rule1.background
        assert_compare_steps(rule1_scenario1.steps, [('given', 'Given', 'scenario step 1', None, None), ('when', 'When', 'scenario step 2', None, None), ('then', 'Then', 'scenario step 3', None, None)])
        assert_compare_steps(rule1_scenario1.all_steps, [('given', 'Given', 'background step 1', None, None), ('when', 'When', 'background step 2', None, None), ('given', 'Given', 'scenario step 1', None, None), ('when', 'When', 'scenario step 2', None, None), ('then', 'Then', 'scenario step 3', None, None)])

    def test_parses_rule_with_tags(self):
        if False:
            for i in range(10):
                print('nop')
        text = u'\nFeature: With Rule\n\n  @rule_tag1\n  @rule_tag2 @rule_tag3\n  Rule: R2\n'.lstrip()
        feature = parse_feature(text)
        rule1 = feature.rules[0]
        assert feature.name == 'With Rule'
        assert feature.background is None
        assert len(feature.rules) == 1
        assert len(feature.scenarios) == 0
        assert rule1.name == 'R2'
        assert rule1.tags == ['rule_tag1', 'rule_tag2', 'rule_tag3']
        assert rule1.description == []
        assert rule1.background is None
        assert len(rule1.scenarios) == 0

    def test_parses_rule_with_description(self):
        if False:
            i = 10
            return i + 15
        text = u'\nFeature: With Rule\n\n  Rule: R3\n    Rule description line 1.\n    Rule description line 2.\n    \n    Rule description line 3.\n'.lstrip()
        feature = parse_feature(text)
        rule1 = feature.rules[0]
        assert feature.name == 'With Rule'
        assert feature.background is None
        assert len(feature.rules) == 1
        assert len(feature.scenarios) == 0
        assert rule1.name == 'R3'
        assert rule1.description == ['Rule description line 1.', 'Rule description line 2.', 'Rule description line 3.']
        assert rule1.tags == []
        assert rule1.background is None
        assert len(rule1.scenarios) == 0

    def test_parses_rule_with_background(self):
        if False:
            print('Hello World!')
        text = u'\nFeature: With Rule\n\n  Rule: R3\n    Background: R3.Background\n      Given background step 1\n      When background step 2\n'.lstrip()
        feature = parse_feature(text)
        rule1 = feature.rules[0]
        assert feature.name == 'With Rule'
        assert feature.background is None
        assert len(feature.rules) == 1
        assert len(feature.scenarios) == 0
        assert rule1.name == 'R3'
        assert rule1.description == []
        assert rule1.tags == []
        assert len(rule1.scenarios) == 0
        assert rule1.background is not None
        assert_compare_steps(rule1.background.steps, [('given', 'Given', 'background step 1', None, None), ('when', 'When', 'background step 2', None, None)])

    def test_parses_rule_without_background_should_inherit_feature_background(self):
        if False:
            for i in range(10):
                print('nop')
        "If a Rule has no Background,\n        it inherits the Feature's Background (if one exists).\n        "
        text = u'\nFeature: With Rule\n  Background: Feature.Background\n    Given feature background step 1\n    When  feature background step 2\n\n  Rule: R3A\n    Scenario: R3A.Scenario_1\n      Given scenario step 1\n      When scenario step 2\n'.lstrip()
        feature = parse_feature(text)
        rule1 = feature.rules[0]
        assert feature.name == 'With Rule'
        assert feature.background is not None
        assert len(feature.rules) == 1
        assert len(feature.scenarios) == 0
        assert rule1.name == 'R3A'
        assert rule1.description == []
        assert rule1.tags == []
        assert len(rule1.scenarios) == 1
        assert rule1.background is not feature.background
        assert rule1.background.inherited_steps == feature.background.steps
        assert list(rule1.background.all_steps) == feature.background.steps
        assert_compare_steps(rule1.scenarios[0].all_steps, [('given', 'Given', 'feature background step 1', None, None), ('when', 'When', 'feature background step 2', None, None), ('given', 'Given', 'scenario step 1', None, None), ('when', 'When', 'scenario step 2', None, None)])

    def test_parses_rule_with_background_inherits_feature_background(self):
        if False:
            for i in range(10):
                print('nop')
        "If a Rule has no Background,\n        it inherits the Feature's Background (if one exists).\n        "
        text = u'\nFeature: With Rule\n  Background: Feature.Background\n    Given feature background step 1\n    When  feature background step 2\n\n  Rule: R3B\n    Background: Rule_R3B.Background\n      Given rule background step 1\n      When  rule background step 2\n\n    Scenario: R3B.Scenario_1\n      Given scenario step 1\n      When scenario step 2\n'.lstrip()
        feature = parse_feature(text)
        rule1 = feature.rules[0]
        assert feature.name == 'With Rule'
        assert feature.background is not None
        assert len(feature.rules) == 1
        assert len(feature.scenarios) == 0
        assert rule1.name == 'R3B'
        assert rule1.description == []
        assert rule1.tags == []
        assert len(rule1.scenarios) == 1
        assert rule1.background is not None
        assert rule1.background is not feature.background
        assert_compare_steps(rule1.scenarios[0].all_steps, [('given', 'Given', 'feature background step 1', None, None), ('when', 'When', 'feature background step 2', None, None), ('given', 'Given', 'rule background step 1', None, None), ('when', 'When', 'rule background step 2', None, None), ('given', 'Given', 'scenario step 1', None, None), ('when', 'When', 'scenario step 2', None, None)])

    def test_parses_rule_with_empty_background_inherits_feature_background(self):
        if False:
            return 10
        'A Rule has empty Background (without any steps) prevents that\n        Feature Background is inherited (if one exists).\n        '
        text = u'\nFeature: With Rule\n  Background: Feature.Background\n    Given feature background step 1\n    When  feature background step 2\n\n  Rule: R3C\n    Background: Rule_R3C.Empty_Background\n\n    Scenario: R3B.Scenario_1\n      Given scenario step 1\n      When scenario step 2\n'.lstrip()
        feature = parse_feature(text)
        rule1 = feature.rules[0]
        assert feature.name == 'With Rule'
        assert feature.background is not None
        assert len(feature.rules) == 1
        assert len(feature.scenarios) == 0
        assert rule1.name == 'R3C'
        assert rule1.description == []
        assert rule1.tags == []
        assert len(rule1.scenarios) == 1
        assert rule1.background is not None
        assert rule1.background is not feature.background
        assert rule1.background.name == 'Rule_R3C.Empty_Background'
        assert_compare_steps(rule1.scenarios[0].all_steps, [('given', 'Given', 'feature background step 1', None, None), ('when', 'When', 'feature background step 2', None, None), ('given', 'Given', 'scenario step 1', None, None), ('when', 'When', 'scenario step 2', None, None)])

    def test_parses_rule_with_scenario(self):
        if False:
            for i in range(10):
                print('nop')
        text = u'\nFeature: With Rule\n\n  Rule: R4\n    Scenario: R4.Scenario_1\n      Given scenario step 1\n      When scenario step 2\n'.lstrip()
        feature = parse_feature(text)
        rule1 = feature.rules[0]
        rule1_scenario1 = rule1.scenarios[0]
        assert feature.name == 'With Rule'
        assert feature.background is None
        assert len(feature.rules) == 1
        assert len(feature.scenarios) == 0
        assert rule1.name == 'R4'
        assert rule1.description == []
        assert rule1.tags == []
        assert rule1.background is None
        assert len(rule1.scenarios) == 1
        assert rule1_scenario1.name == 'R4.Scenario_1'
        assert_compare_steps(rule1_scenario1.steps, [('given', 'Given', 'scenario step 1', None, None), ('when', 'When', 'scenario step 2', None, None)])

    def test_parses_rule_with_two_scenarios(self):
        if False:
            i = 10
            return i + 15
        text = u'\nFeature: With Rule\n\n  Rule: R4\n    Scenario: R4.Scenario_1\n      Given scenario 1 step 1\n      When scenario 1 step 2\n\n    Scenario: R4.Scenario_2\n      Given scenario 2 step 1\n      When scenario 2 step 2\n'.lstrip()
        feature = parse_feature(text)
        rule1 = feature.rules[0]
        rule1_scenario1 = rule1.scenarios[0]
        rule1_scenario2 = rule1.scenarios[1]
        assert feature.name == 'With Rule'
        assert feature.background is None
        assert len(feature.rules) == 1
        assert len(feature.scenarios) == 0
        assert rule1.name == 'R4'
        assert rule1.description == []
        assert rule1.tags == []
        assert rule1.background is None
        assert len(rule1.scenarios) == 2
        assert rule1_scenario1.name == 'R4.Scenario_1'
        assert rule1_scenario1.parent is rule1
        assert rule1_scenario1.feature is feature
        assert_compare_steps(rule1_scenario1.steps, [('given', 'Given', 'scenario 1 step 1', None, None), ('when', 'When', 'scenario 1 step 2', None, None)])
        assert rule1_scenario2.name == 'R4.Scenario_2'
        assert rule1_scenario2.parent is rule1
        assert rule1_scenario2.feature is feature
        assert_compare_steps(rule1_scenario2.steps, [('given', 'Given', 'scenario 2 step 1', None, None), ('when', 'When', 'scenario 2 step 2', None, None)])

    def test_parses_rule_with_scenario_outline(self):
        if False:
            for i in range(10):
                print('nop')
        text = u'\nFeature: With Rule\n\n  Rule: R5\n    Scenario Outline: R5.ScenarioOutline\n      Given step with name "<name>"\n      When step uses "<param2>"\n      \n      Examples:\n        | name  | param2 |\n        | Alice | 1      |\n        | Bob   | 2      |\n'.lstrip()
        feature = parse_feature(text)
        rule1 = feature.rules[0]
        rule1_scenario1 = rule1.scenarios[0]
        assert feature.name == 'With Rule'
        assert feature.background is None
        assert len(feature.rules) == 1
        assert len(feature.scenarios) == 0
        assert rule1.name == 'R5'
        assert rule1.description == []
        assert rule1.tags == []
        assert rule1.background is None
        assert len(rule1.scenarios) == 1
        assert len(rule1.scenarios[0].scenarios) == 2
        assert_compare_steps(rule1_scenario1.scenarios[0].steps, [('given', 'Given', 'step with name "Alice"', None, None), ('when', 'When', 'step uses "1"', None, None)])
        assert_compare_steps(rule1_scenario1.scenarios[1].steps, [('given', 'Given', 'step with name "Bob"', None, None), ('when', 'When', 'step uses "2"', None, None)])

    def test_parses_rule_with_two_scenario_outlines(self):
        if False:
            i = 10
            return i + 15
        text = u'\nFeature: With Rule\n\n  Rule: R5\n    Scenario Outline: R5.ScenarioOutline_1\n      Given step with name "<name>"\n      When step uses "<param2>"\n\n      Examples:\n        | name  | param2 |\n        | Alice | 1      |\n        | Bob   | 2      |\n\n    Scenario Outline: R5.ScenarioOutline_2\n      Given step with name "<name>"\n\n      Examples:\n        | name    |\n        | Charly  |\n        | Dorothy |\n'.lstrip()
        feature = parse_feature(text)
        rule1 = feature.rules[0]
        rule1_scenario1 = rule1.scenarios[0]
        rule1_scenario2 = rule1.scenarios[1]
        assert feature.name == 'With Rule'
        assert feature.background is None
        assert len(feature.rules) == 1
        assert len(feature.scenarios) == 0
        assert rule1.name == 'R5'
        assert rule1.description == []
        assert rule1.tags == []
        assert rule1.background is None
        assert len(rule1.scenarios) == 2
        assert len(rule1_scenario1.scenarios) == 2
        assert rule1_scenario1.scenarios[0].name == 'R5.ScenarioOutline_1 -- @1.1 '
        assert rule1_scenario1.scenarios[0].parent is rule1_scenario1
        assert rule1_scenario1.scenarios[0].feature is feature
        assert_compare_steps(rule1_scenario1.scenarios[0].steps, [('given', 'Given', 'step with name "Alice"', None, None), ('when', 'When', 'step uses "1"', None, None)])
        assert rule1_scenario1.scenarios[1].name == 'R5.ScenarioOutline_1 -- @1.2 '
        assert rule1_scenario1.scenarios[0].parent is rule1_scenario1
        assert rule1_scenario1.scenarios[0].feature is feature
        assert_compare_steps(rule1_scenario1.scenarios[1].steps, [('given', 'Given', 'step with name "Bob"', None, None), ('when', 'When', 'step uses "2"', None, None)])
        assert len(rule1_scenario2.scenarios) == 2
        assert rule1_scenario2.scenarios[0].name == 'R5.ScenarioOutline_2 -- @1.1 '
        assert rule1_scenario2.scenarios[0].parent is rule1_scenario2
        assert rule1_scenario2.scenarios[0].feature is feature
        assert_compare_steps(rule1_scenario2.scenarios[0].steps, [('given', 'Given', 'step with name "Charly"', None, None)])
        assert rule1_scenario2.scenarios[1].name == 'R5.ScenarioOutline_2 -- @1.2 '
        assert rule1_scenario2.scenarios[1].parent is rule1_scenario2
        assert rule1_scenario2.scenarios[1].feature is feature
        assert_compare_steps(rule1_scenario2.scenarios[1].steps, [('given', 'Given', 'step with name "Dorothy"', None, None)])

    def test_parses_two_rules(self):
        if False:
            i = 10
            return i + 15
        text = u'\nFeature: With Rule\n\n  Rule: R1\n    Scenario: R1.Scenario_1\n      Given scenario 1 step 1\n      When scenario 1 step 2\n  \n  Rule: R2\n    Scenario Outline: R2.ScenarioOutline_1\n      Given step with name "<name>"\n      When step uses "<param2>"\n      \n      Examples:\n        | name  | param2 |\n        | Alice | 1      |\n        | Bob   | 2      |\n'.lstrip()
        feature = parse_feature(text)
        rule1 = feature.rules[0]
        rule2 = feature.rules[1]
        rule1_scenario1 = rule1.scenarios[0]
        rule2_scenario1 = rule2.scenarios[0]
        assert feature.name == 'With Rule'
        assert feature.background is None
        assert len(feature.rules) == 2
        assert len(feature.scenarios) == 0
        assert rule1.name == 'R1'
        assert rule1.parent is feature
        assert rule1.feature is feature
        assert rule1.description == []
        assert rule1.tags == []
        assert rule1.background is None
        assert len(rule1.scenarios) == 1
        assert rule1_scenario1.name == 'R1.Scenario_1'
        assert rule1_scenario1.parent is rule1
        assert rule1_scenario1.feature is feature
        assert_compare_steps(rule1_scenario1.steps, [('given', 'Given', 'scenario 1 step 1', None, None), ('when', 'When', 'scenario 1 step 2', None, None)])
        assert rule2.name == 'R2'
        assert rule2.parent is feature
        assert rule2.feature is feature
        assert rule2.description == []
        assert rule2.tags == []
        assert rule2.background is None
        assert len(rule2.scenarios) == 1
        assert len(rule2.scenarios[0].scenarios) == 2
        assert rule2_scenario1.scenarios[0].name == 'R2.ScenarioOutline_1 -- @1.1 '
        assert rule2_scenario1.scenarios[0].parent is rule2_scenario1
        assert rule2_scenario1.scenarios[0].feature is feature
        assert_compare_steps(rule2_scenario1.scenarios[0].steps, [('given', 'Given', 'step with name "Alice"', None, None), ('when', 'When', 'step uses "1"', None, None)])
        assert rule2_scenario1.scenarios[1].name == 'R2.ScenarioOutline_1 -- @1.2 '
        assert rule2_scenario1.scenarios[1].parent is rule2_scenario1
        assert rule2_scenario1.scenarios[1].feature is feature
        assert_compare_steps(rule2_scenario1.scenarios[1].steps, [('given', 'Given', 'step with name "Bob"', None, None), ('when', 'When', 'step uses "2"', None, None)])

    def test_parse_background_scenario_and_rules(self):
        if False:
            i = 10
            return i + 15
        'HINT: Some Scenarios may exist before the first Rule.'
        text = u'\nFeature: With Scenarios and Rules\n\n  Background: Feature.Background\n    Given feature background step_1\n    When  feature background step_2\n\n  Scenario: Scenario_1\n    Given scenario_1 step_1\n    When  scenario_1 step_2\n    \n  Rule: R1\n    Background: R1.Background\n      Given rule R1 background step_1\n      \n    Scenario: R1.Scenario_1\n      Given rule R1 scenario_1 step_1\n      When  rule R1 scenario_1 step_2\n\n  Rule: R2\n    Scenario: R2.Scenario_1\n      Given rule R2 scenario_1 step_1\n      When  rule R2 scenario_1 step_2\n'.lstrip()
        feature = parse_feature(text)
        assert feature.name == 'With Scenarios and Rules'
        assert feature.background is not None
        assert len(feature.scenarios) == 1
        assert len(feature.rules) == 2
        assert len(feature.run_items) == 3
        scenario1 = feature.scenarios[0]
        rule1 = feature.rules[0]
        rule2 = feature.rules[1]
        rule1_scenario1 = rule1.scenarios[0]
        rule2_scenario1 = rule2.scenarios[0]
        assert feature.run_items == [scenario1, rule1, rule2]
        assert scenario1.name == 'Scenario_1'
        assert scenario1.background is feature.background
        assert scenario1.parent is feature
        assert scenario1.feature is feature
        assert scenario1.tags == []
        assert scenario1.description == []
        assert_compare_steps(scenario1.all_steps, [(u'given', u'Given', u'feature background step_1', None, None), (u'when', u'When', u'feature background step_2', None, None), (u'given', u'Given', u'scenario_1 step_1', None, None), (u'when', u'When', u'scenario_1 step_2', None, None)])
        assert rule1.name == 'R1'
        assert rule1.parent is feature
        assert rule1.feature is feature
        assert rule1.description == []
        assert rule1.tags == []
        assert rule1.background is not None
        assert len(rule1.scenarios) == 1
        assert rule1_scenario1.name == 'R1.Scenario_1'
        assert rule1_scenario1.parent is rule1
        assert rule1_scenario1.feature is feature
        assert_compare_steps(rule1_scenario1.all_steps, [('given', 'Given', 'feature background step_1', None, None), ('when', 'When', 'feature background step_2', None, None), ('given', 'Given', 'rule R1 background step_1', None, None), ('given', 'Given', 'rule R1 scenario_1 step_1', None, None), ('when', 'When', 'rule R1 scenario_1 step_2', None, None)])
        assert rule2.name == 'R2'
        assert rule2.parent is feature
        assert rule2.feature is feature
        assert rule2.description == []
        assert rule2.tags == []
        assert rule2.background is not feature.background
        assert list(rule2.background.inherited_steps) == list(feature.background.steps)
        assert list(rule2.background.all_steps) == list(feature.background.steps)
        assert len(rule2.scenarios) == 1
        assert rule2_scenario1.name == 'R2.Scenario_1'
        assert rule2_scenario1.parent is rule2
        assert rule2_scenario1.feature is feature
        assert_compare_steps(rule2_scenario1.all_steps, [('given', 'Given', 'feature background step_1', None, None), ('when', 'When', 'feature background step_2', None, None), ('given', 'Given', 'rule R2 scenario_1 step_1', None, None), ('when', 'When', 'rule R2 scenario_1 step_2', None, None)])

class TestParser4Background(object):
    """Verify feature.background to rule.background inheritance, etc."""

    def test_parse__norule_scenarios_use_feature_background(self):
        if False:
            for i in range(10):
                print('nop')
        'AFFECTED: Scenarios outside of rules (before first rule).'
        text = u'\n            Feature: With Scenarios and Rules\n            \n              Background: Feature.Background\n                Given feature background step_1\n            \n              Scenario: Scenario_1\n                Given scenario_1 step_1\n            \n              Rule: R1\n                Scenario: R1.Scenario_1\n                  Given rule R1 scenario_1 step_1\n            '.lstrip()
        feature = parse_feature(text)
        assert feature.name == 'With Scenarios and Rules'
        assert feature.background is not None
        assert len(feature.scenarios) == 1
        assert len(feature.rules) == 1
        assert len(feature.run_items) == 2
        scenario1 = feature.scenarios[0]
        rule1 = feature.rules[0]
        assert feature.run_items == [scenario1, rule1]
        assert scenario1.name == 'Scenario_1'
        assert scenario1.background is feature.background
        assert scenario1.background_steps == feature.background.steps
        assert_compare_steps(scenario1.all_steps, [(u'given', u'Given', u'feature background step_1', None, None), (u'given', u'Given', u'scenario_1 step_1', None, None)])

    def test_parse__norule_scenarios_with_disabled_background(self):
        if False:
            print('Hello World!')
        'AFFECTED: Scenarios outside of rules (before first rule).'
        text = u'\n            Feature: Scenario with disabled background\n            \n              Background: Feature.Background\n                Given feature background step_1\n            \n              @fixture.behave.disable_background\n              Scenario: Scenario_1\n                Given scenario_1 step_1\n            \n              Scenario: Scenario_2\n                Given scenario_2 step_1\n            '.lstrip()
        feature = parse_feature(text)
        assert feature.name == 'Scenario with disabled background'
        assert feature.background is not None
        assert len(feature.scenarios) == 2
        assert len(feature.run_items) == 2
        scenario1 = feature.scenarios[0]
        scenario2 = feature.scenarios[1]
        assert feature.run_items == [scenario1, scenario2]
        scenario1.use_background = False
        assert scenario1.name == 'Scenario_1'
        assert scenario1.background is feature.background
        assert scenario1.background_steps != feature.background.steps
        assert scenario1.background_steps == []
        assert_compare_steps(scenario1.all_steps, [(u'given', u'Given', u'scenario_1 step_1', None, None)])
        assert scenario2.name == 'Scenario_2'
        assert scenario2.background is feature.background
        assert scenario2.background_steps == feature.background.steps
        assert_compare_steps(scenario2.all_steps, [(u'given', u'Given', u'feature background step_1', None, None), (u'given', u'Given', u'scenario_2 step_1', None, None)])

    def test_parse__rule_scenarios_inherit_feature_background_without_rule_background(self):
        if False:
            while True:
                i = 10
        text = u'\n            Feature: With Background and Rule\n    \n              Background: Feature.Background\n                Given feature background step_1\n    \n              Rule: R1\n                Scenario: R1.Scenario_1\n                  Given rule R1 scenario_1 step_1\n            '.lstrip()
        feature = parse_feature(text)
        assert feature.name == 'With Background and Rule'
        assert feature.background is not None
        assert len(feature.scenarios) == 0
        assert len(feature.rules) == 1
        assert len(feature.run_items) == 1
        rule1 = feature.rules[0]
        rule1_scenario1 = rule1.scenarios[0]
        assert feature.run_items == [rule1]
        assert rule1_scenario1.name == 'R1.Scenario_1'
        assert rule1_scenario1.background is not None
        assert rule1_scenario1.background_steps == feature.background.steps
        assert_compare_steps(rule1_scenario1.all_steps, [(u'given', u'Given', u'feature background step_1', None, None), (u'given', u'Given', u'rule R1 scenario_1 step_1', None, None)])

    def test_parse__rule_scenarios_inherit_feature_background_with_rule_background(self):
        if False:
            while True:
                i = 10
        text = u'\n            Feature: With Feature.Background and Rule.Background\n\n              Background: Feature.Background\n                Given feature background step_1\n\n              Rule: R1\n                Background: R1.Background\n                  Given rule R1 background step_1\n                \n                Scenario: R1.Scenario_1\n                  Given rule R1 scenario_1 step_1\n            '.lstrip()
        feature = parse_feature(text)
        assert feature.name == 'With Feature.Background and Rule.Background'
        assert feature.background is not None
        assert len(feature.scenarios) == 0
        assert len(feature.rules) == 1
        assert len(feature.run_items) == 1
        rule1 = feature.rules[0]
        rule1_scenario1 = rule1.scenarios[0]
        assert feature.run_items == [rule1]
        assert rule1.background is not None
        assert rule1.background is not feature.background
        assert rule1.background.inherited_steps == feature.background.steps
        assert list(rule1.background.all_steps) != feature.background.steps
        assert rule1_scenario1.name == 'R1.Scenario_1'
        assert rule1_scenario1.background is rule1.background
        assert rule1_scenario1.background_steps == list(rule1.background.all_steps)
        assert_compare_steps(rule1_scenario1.all_steps, [(u'given', u'Given', u'feature background step_1', None, None), (u'given', u'Given', u'rule R1 background step_1', None, None), (u'given', u'Given', u'rule R1 scenario_1 step_1', None, None)])

    def test_parse__rule_scenarios_with_rule_background_when_background_inheritance_is_disabled(self):
        if False:
            while True:
                i = 10
        text = u'\n            Feature: With Feature Background Inheritance disabled\n\n              Background: Feature.Background\n                Given feature background step_1\n\n              @fixture.behave.override_background\n              Rule: R1\n                Background: R1.Background\n                  Given rule R1 background step_1\n\n                Scenario: R1.Scenario_1\n                  Given rule R1 scenario_1 step_1\n            '.lstrip()
        feature = parse_feature(text)
        assert feature.name == 'With Feature Background Inheritance disabled'
        assert feature.background is not None
        assert len(feature.scenarios) == 0
        assert len(feature.rules) == 1
        assert len(feature.run_items) == 1
        rule1 = feature.rules[0]
        rule1_scenario1 = rule1.scenarios[0]
        assert feature.run_items == [rule1]
        rule1.use_background_inheritance = False
        assert rule1.background is not None
        assert rule1.background.use_inheritance is False
        assert rule1.background is not feature.background
        assert rule1.background.inherited_steps == []
        assert rule1.background.inherited_steps != feature.background.steps
        assert list(rule1.background.all_steps) != feature.background.steps
        assert rule1_scenario1.name == 'R1.Scenario_1'
        assert rule1_scenario1.background is rule1.background
        assert rule1_scenario1.background_steps == rule1.background.steps
        assert rule1_scenario1.background_steps == list(rule1.background.all_steps)
        assert_compare_steps(rule1_scenario1.all_steps, [(u'given', u'Given', u'rule R1 background step_1', None, None), (u'given', u'Given', u'rule R1 scenario_1 step_1', None, None)])

    def test_parse__rule_scenarios_without_rule_background_when_background_inheritance_is_disabled_without(self):
        if False:
            for i in range(10):
                print('nop')
        text = u'\n            Feature: With Feature Background Inheritance disabled\n\n              Background: Feature.Background\n                Given feature background step_1\n\n              @fixture.behave.override_background\n              Rule: R1\n                Scenario: R1.Scenario_1\n                  Given rule R1 scenario_1 step_1\n            '.lstrip()
        feature = parse_feature(text)
        assert feature.name == 'With Feature Background Inheritance disabled'
        assert feature.background is not None
        assert len(feature.scenarios) == 0
        assert len(feature.rules) == 1
        assert len(feature.run_items) == 1
        rule1 = feature.rules[0]
        rule1_scenario1 = rule1.scenarios[0]
        assert feature.run_items == [rule1]
        rule1.use_background_inheritance = False
        assert rule1.background is not None
        assert rule1.background.use_inheritance is False
        assert rule1.background is not feature.background
        assert rule1.background.inherited_steps == []
        assert rule1_scenario1.name == 'R1.Scenario_1'
        assert rule1_scenario1.background is rule1.background
        assert rule1_scenario1.background_steps == rule1.background.steps
        assert rule1_scenario1.background_steps == list(rule1.background.all_steps)
        assert_compare_steps(rule1_scenario1.all_steps, [(u'given', u'Given', u'rule R1 scenario_1 step_1', None, None)])

    def test_parse__rule_scenarios_without_feature_background_and_with_rule_background(self):
        if False:
            for i in range(10):
                print('nop')
        text = u'\n            Feature: Without Feature.Background and with Rule.Background\n\n              Rule: R1\n                Background: R1.Background\n                  Given rule R1 background step_1\n\n                Scenario: R1.Scenario_1\n                  Given rule R1 scenario_1 step_1\n            '.lstrip()
        feature = parse_feature(text)
        assert feature.name == 'Without Feature.Background and with Rule.Background'
        assert feature.background is None
        assert len(feature.scenarios) == 0
        assert len(feature.rules) == 1
        assert len(feature.run_items) == 1
        rule1 = feature.rules[0]
        rule1_scenario1 = rule1.scenarios[0]
        assert feature.run_items == [rule1]
        assert rule1.background is not None
        assert rule1.background is not feature.background
        assert rule1.background.inherited_steps == []
        assert rule1_scenario1.name == 'R1.Scenario_1'
        assert rule1_scenario1.background is rule1.background
        assert rule1_scenario1.background_steps == rule1.background.steps
        assert rule1_scenario1.background_steps == list(rule1.background.all_steps)
        assert_compare_steps(rule1_scenario1.all_steps, [(u'given', u'Given', u'rule R1 background step_1', None, None), (u'given', u'Given', u'rule R1 scenario_1 step_1', None, None)])

    def test_parse__rule_scenarios_without_feature_and_rule_background(self):
        if False:
            return 10
        text = u'\n            Feature: Without Feature.Background and Rule.Background\n\n              Rule: R1\n                Scenario: R1.Scenario_1\n                  Given rule R1 scenario_1 step_1\n            '.lstrip()
        feature = parse_feature(text)
        assert feature.name == 'Without Feature.Background and Rule.Background'
        assert feature.background is None
        assert len(feature.scenarios) == 0
        assert len(feature.rules) == 1
        assert len(feature.run_items) == 1
        rule1 = feature.rules[0]
        rule1_scenario1 = rule1.scenarios[0]
        assert feature.run_items == [rule1]
        assert rule1.background is None
        assert rule1_scenario1.name == 'R1.Scenario_1'
        assert rule1_scenario1.background is None
        assert rule1_scenario1.background is rule1.background
        assert rule1_scenario1.background_steps == []
        assert_compare_steps(rule1_scenario1.all_steps, [(u'given', u'Given', u'rule R1 scenario_1 step_1', None, None)])

class TestParser4Scenario(object):

    def test_use_example_alias(self):
        if False:
            i = 10
            return i + 15
        'HINT: Some Scenarios may exist before the first Rule.'
        text = u'\nFeature: With Example as Alias for Scenario\n\n  Example: Scenario_1\n    Given scenario_1 step_1\n    When  scenario_1 step_2\n'.lstrip()
        feature = parse_feature(text)
        assert feature.name == 'With Example as Alias for Scenario'
        assert len(feature.scenarios) == 1
        assert len(feature.run_items) == 1
        scenario1 = feature.scenarios[0]
        assert feature.run_items == [scenario1]
        assert scenario1.name == 'Scenario_1'
        assert scenario1.keyword == 'Example'
        assert scenario1.background is None
        assert scenario1.parent is feature
        assert scenario1.feature is feature
        assert scenario1.tags == []
        assert scenario1.description == []
        assert_compare_steps(scenario1.all_steps, [('given', 'Given', 'scenario_1 step_1', None, None), ('when', 'When', 'scenario_1 step_2', None, None)])

class TestParser4ScenarioOutline(object):

    def test_use_scenario_template_alias(self):
        if False:
            i = 10
            return i + 15
        'HINT: Some Scenarios may exist before the first Rule.'
        text = u'\n    Feature: Use ScenarioTemplate as Alias for ScenarioOutline\n\n      Scenario Template: ScenarioOutline_1\n        Given a step with name "<name>"\n        \n        Examples:\n          | name  |\n          | Alice |\n          | Bob   |\n    '.lstrip()
        feature = parse_feature(text)
        assert feature.name == 'Use ScenarioTemplate as Alias for ScenarioOutline'
        assert len(feature.scenarios) == 1
        assert len(feature.run_items) == 1
        scenario1 = feature.scenarios[0]
        assert feature.run_items == [scenario1]
        assert scenario1.name == 'ScenarioOutline_1'
        assert scenario1.keyword == 'Scenario Template'
        assert scenario1.background is None
        assert scenario1.parent is feature
        assert scenario1.feature is feature
        assert scenario1.tags == []
        assert scenario1.description == []
        assert len(scenario1.scenarios) == 2
        assert scenario1.scenarios[0].name == 'ScenarioOutline_1 -- @1.1 '
        assert_compare_steps(scenario1.scenarios[0].steps, [('given', 'Given', 'a step with name "Alice"', None, None)])
        assert scenario1.scenarios[1].name == 'ScenarioOutline_1 -- @1.2 '
        assert_compare_steps(scenario1.scenarios[1].steps, [('given', 'Given', 'a step with name "Bob"', None, None)])