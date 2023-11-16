"""
Provides step definitions that test tag logic for selected features, scenarios.

.. code-block:: Gherkin

    # -- Scenario: Select scenarios with tags
    Given I use the behave model builder with:
        | statement  | name   | tags      | Comment |
        | Scenario   | A1     | @foo      |          |
        | Scenario   | A3     | @foo @bar |          |
        | Scenario   | B3     |           | Untagged |
    When I run the behave with tags
    Then the following scenarios are selected with cmdline:
        | cmdline                    | selected           | Logic comment |
        | --tags=@foo                | A1, A3, B2         | @foo          |
        | --tags=-@foo               | A1, A3, B2         | @foo          |

.. code-block:: Gherkin

    # IDEA:
    # -- Scenario: Select scenarios with tags
    Given I use the behave model builder with:
        | statement  | name   | tags      | Comment |
        | Feature    | Alice  | @alice    |          |
        | Scenario   | A1     | @foo      |          |
        | Scenario   | A2     | @bar      |          |
        | Scenario   | A3     | @foo @bar |          |
        | Feature    | Bob    | @bob      |          |
        | Scenario   | B1     | @bar      |          |
        | Scenario   | B2     | @foo      |          |
        | Scenario   | B3     |           | Untagged |
    When I run the behave with options "--tags=@foo"
    Then the following scenarios are selected:
        | statement  | name   | selected  |
        | Scenario   | A1     | yes       |
        | Scenario   | A2     | no        |
        | Scenario   | A3     | yes       |
        | Scenario   | B1     | no        |
        | Scenario   | B2     | yes       |
        | Scenario   | B3     | no        |
"""
from __future__ import absolute_import
from behave import given, when, then
from behave_model_util import BehaveModelBuilder, convert_comma_list
from behave_model_util import run_model_with_cmdline, collect_selected_and_skipped_scenarios
from hamcrest import assert_that, equal_to

@given('a behave model with')
def step_given_a_behave_model_with_table(context):
    if False:
        i = 10
        return i + 15
    '\n    Build a behave feature model from a tabular description.\n\n    .. code-block:: gherkin\n\n        # -- Scenario: Select scenarios with tags\n        Given I use the behave model builder with:\n            | statement  | name   | tags      | Comment |\n            | Scenario   | S0     |           | Untagged |\n            | Scenario   | S1     | @foo      |          |\n            | Scenario   | S3     | @foo @bar |          |\n    '
    assert context.table, 'REQUIRE: context.table'
    context.table.require_columns(BehaveModelBuilder.REQUIRED_COLUMNS)
    model_builder = BehaveModelBuilder()
    context.behave_model = model_builder.build_model_from_table(context.table)

@when('I run the behave model with "{hint}"')
def step_when_run_behave_model_with_hint(context, hint):
    if False:
        i = 10
        return i + 15
    pass

@then('the following scenarios are selected with cmdline')
def step_then_scenarios_are_selected_with_cmdline(context):
    if False:
        i = 10
        return i + 15
    '\n    .. code-block:: Gherkin\n\n        Then the following scenarios are selected with cmdline:\n            | cmdline      | selected?    | Logic comment |\n            | --tags=@foo  | A1, A3, B2   | @foo          |\n    '
    assert context.behave_model, 'REQUIRE: context attribute'
    assert context.table, 'REQUIRE: context.table'
    context.table.require_columns(['cmdline', 'selected?'])
    model = context.behave_model
    for (row_index, row) in enumerate(context.table.rows):
        cmdline = row['cmdline']
        expected_selected_names = convert_comma_list(row['selected?'])
        run_model_with_cmdline(model, cmdline)
        (selected, skipped) = collect_selected_and_skipped_scenarios(model)
        actual_selected = [scenario.name for scenario in selected]
        assert_that(actual_selected, equal_to(expected_selected_names), 'cmdline=%s (row=%s)' % (cmdline, row_index))