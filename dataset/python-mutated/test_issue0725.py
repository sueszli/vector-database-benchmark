"""
https://github.com/behave/behave/issues/725

ANALYSIS:
----------

ScenarioOutlineBuilder did not copy ScenarioOutline.description
to the Scenarios that were created from the ScenarioOutline.
"""
from __future__ import absolute_import, print_function
from behave.parser import parse_feature

def test_issue():
    if False:
        print('Hello World!')
    'Verifies that issue #725 is fixed.'
    text = u'\nFeature: ScenarioOutline with description\n\n  Scenario Outline: SO_1\n    Description line 1 for ScenarioOutline.\n    Description line 2 for ScenarioOutline.\n\n    Given a step with "<name>"\n    \n    Examples:\n      | name  |\n      | Alice |\n      | Bob   |\n'.lstrip()
    feature = parse_feature(text)
    assert len(feature.scenarios) == 1
    scenario_outline_1 = feature.scenarios[0]
    assert len(scenario_outline_1.scenarios) == 2
    expected_description = ['Description line 1 for ScenarioOutline.', 'Description line 2 for ScenarioOutline.']
    assert scenario_outline_1.description == expected_description
    assert scenario_outline_1.scenarios[0].description == expected_description
    assert scenario_outline_1.scenarios[1].description == expected_description