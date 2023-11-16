'''
Provides step definitions that test how the behave runner selects feature files.

EXAMPLE:

.. code-block:: gherkin

    Given behave has the following feature fileset:
      """
      features/alice.feature
      features/bob.feature
      features/barbi.feature
      """
    When behave includes feature files with "features/a.*\\.feature"
    And  behave excludes feature files with "features/b.*\\.feature"
    Then the following feature files are selected:
      """
      features/alice.feature
      """
'''
from __future__ import absolute_import
from copy import copy
import re
import six
from hamcrest import assert_that, equal_to
from behave import given, when, then
from behave.runner_util import FeatureListParser

class BasicBehaveRunner(object):

    def __init__(self, config=None):
        if False:
            print('Hello World!')
        self.config = config
        self.feature_files = []

    def select_files(self):
        if False:
            while True:
                i = 10
        '\n        Emulate behave runners file selection by using include/exclude patterns.\n        :return: List of selected feature filenames.\n        '
        selected = []
        for filename in self.feature_files:
            if not self.config.exclude(filename):
                selected.append(six.text_type(filename))
        return selected

@given('behave has the following feature fileset')
def step_given_behave_has_feature_fileset(context):
    if False:
        print('Hello World!')
    assert context.text is not None, 'REQUIRE: text'
    behave_runner = BasicBehaveRunner(config=copy(context.config))
    behave_runner.feature_files = FeatureListParser.parse(context.text)
    context.behave_runner = behave_runner

@when('behave includes all feature files')
def step_when_behave_includes_all_feature_files(context):
    if False:
        return 10
    assert context.behave_runner, 'REQUIRE: context.behave_runner'
    context.behave_runner.config.include_re = None

@when('behave includes feature files with "{pattern}"')
def step_when_behave_includes_feature_files_with_pattern(context, pattern):
    if False:
        print('Hello World!')
    assert context.behave_runner, 'REQUIRE: context.behave_runner'
    context.behave_runner.config.include_re = re.compile(pattern)

@when('behave excludes no feature files')
def step_when_behave_excludes_no_feature_files(context):
    if False:
        while True:
            i = 10
    assert context.behave_runner, 'REQUIRE: context.behave_runner'
    context.behave_runner.config.exclude_re = None

@when('behave excludes feature files with "{pattern}"')
def step_when_behave_excludes_feature_files_with_pattern(context, pattern):
    if False:
        i = 10
        return i + 15
    assert context.behave_runner, 'REQUIRE: context.behave_runner'
    context.behave_runner.config.exclude_re = re.compile(pattern)

@then('the following feature files are selected')
def step_then_feature_files_are_selected_with_text(context):
    if False:
        i = 10
        return i + 15
    assert context.text is not None, 'REQUIRE: text'
    assert context.behave_runner, 'REQUIRE: context.behave_runner'
    selected_files = context.text.strip().splitlines()
    actual_files = context.behave_runner.select_files()
    assert_that(actual_files, equal_to(selected_files))