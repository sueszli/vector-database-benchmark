"""suites tests"""
import random
from hamcrest import all_of, assert_that, calling, equal_to, has_entry, has_length, instance_of, is_, raises
from deepchecks import __version__
from deepchecks.core import CheckFailure, CheckResult, ConditionCategory, ConditionResult, SuiteResult
from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError
from deepchecks.core.suite import BaseSuite
from deepchecks.tabular import Dataset, SingleDatasetCheck, Suite, TrainTestCheck
from deepchecks.tabular import checks as tabular_checks
from deepchecks.tabular import datasets
from deepchecks.tabular.suites import data_integrity, model_evaluation

class SimpleDatasetCheck(SingleDatasetCheck):

    def run_logic(self, context, dataset_kind) -> CheckResult:
        if False:
            print('Hello World!')
        return CheckResult('Simple Check')

class SimpleTwoDatasetsCheck(TrainTestCheck):

    def run_logic(self, context) -> CheckResult:
        if False:
            i = 10
            return i + 15
        return CheckResult('Simple Check')

def test_suite_instantiation_with_incorrect_args():
    if False:
        while True:
            i = 10
    incorrect_check_suite_args = ('test suite', SimpleDatasetCheck(), object())
    assert_that(calling(Suite).with_args(*incorrect_check_suite_args), raises(DeepchecksValueError))

def test_run_suite_with_incorrect_args():
    if False:
        return 10
    suite = Suite('test suite', SimpleDatasetCheck(), SimpleTwoDatasetsCheck())
    args = {'train_dataset': None, 'test_dataset': None}
    assert_that(calling(suite.run).with_args(**args), raises(DeepchecksValueError, 'At least one dataset \\(or model\\) must be passed to the method!'))

def test_select_results_with_and_without_args_from_suite_result():
    if False:
        for i in range(10):
            print('nop')
    result1 = CheckResult(0, 'check1')
    result1.conditions_results = [ConditionResult(ConditionCategory.PASS)]
    result2 = CheckResult(0, 'check2')
    result2.conditions_results = [ConditionResult(ConditionCategory.FAIL)]
    args = {'idx': [1, 2], 'names': ['check1', 'check2']}
    assert_that(calling(SuiteResult('test', [result1]).select_results).with_args(), raises(DeepchecksNotSupportedError, 'Either idx or names should be passed'))
    assert_that(calling(SuiteResult('test', [result1, result2]).select_results).with_args(**args), raises(DeepchecksNotSupportedError, 'Only one of idx or names should be passed'))

def test_select_results_with_indexes_and_names_from_suite_result():
    if False:
        print('Hello World!')
    data = datasets.regression.avocado.load_data(data_format='DataFrame', as_train_test=False)
    ds = Dataset(data, cat_features=['type'], datetime_name='Date', label='AveragePrice')
    integ_suite = data_integrity()
    suite_result = integ_suite.run(ds)
    suite_results_by_indexes = suite_result.select_results(idx=[0, 2])
    suite_results_by_name = suite_result.select_results(names=['Conflicting Labels - Train Dataset', 'Outlier Sample Detection', 'mixed_Nulls'])
    assert_that(len(suite_results_by_indexes), equal_to(2))
    assert_that(len(suite_results_by_name), equal_to(3))
    assert_that(suite_results_by_indexes[0].get_header(), equal_to('Feature-Feature Correlation'))
    assert_that(suite_results_by_indexes[1].get_header(), equal_to('Single Value in Column'))
    assert_that(suite_results_by_name[0].get_header(), equal_to('Conflicting Labels - Train Dataset'))
    assert_that(suite_results_by_name[1].get_header(), equal_to('Outlier Sample Detection'))
    assert_that(suite_results_by_name[2].get_header(), equal_to('Mixed Nulls'))

def test_add_check_to_the_suite():
    if False:
        print('Hello World!')
    number_of_checks = random.randint(0, 50)

    def produce_checks(count):
        if False:
            return 10
        return [SimpleDatasetCheck() for _ in range(count)]
    first_suite = Suite('first suite')
    second_suite = Suite('second suite')
    assert_that(len(first_suite.checks), equal_to(0))
    assert_that(len(second_suite.checks), equal_to(0))
    for check in produce_checks(number_of_checks):
        first_suite.add(check)
    assert_that(len(first_suite.checks), equal_to(number_of_checks))
    second_suite.add(first_suite)
    assert_that(len(second_suite.checks), equal_to(number_of_checks))

def test_try_add_not_a_check_to_the_suite():
    if False:
        while True:
            i = 10
    suite = Suite('second suite')
    assert_that(calling(suite.add).with_args(object()), raises(DeepchecksValueError, 'Suite received unsupported object type: object'))

def test_try_add_check_suite_to_itself():
    if False:
        i = 10
        return i + 15
    suite = Suite('second suite', SimpleDatasetCheck(), SimpleTwoDatasetsCheck())
    assert_that(len(suite.checks), equal_to(2))
    suite.add(suite)
    assert_that(len(suite.checks), equal_to(2))

def test_suite_static_indexes():
    if False:
        while True:
            i = 10
    first_check = SimpleDatasetCheck()
    second_check = SimpleTwoDatasetsCheck()
    suite = Suite('first suite', first_check, second_check)
    assert_that(len(suite.checks), equal_to(2))
    assert_that(suite[1], is_(second_check))
    assert_that(suite[0], is_(first_check))
    suite.remove(0)
    assert_that(len(suite.checks), equal_to(1))
    assert_that(suite[1], is_(second_check))

def test_access_removed_check_by_index():
    if False:
        i = 10
        return i + 15
    first_check = SimpleDatasetCheck()
    second_check = SimpleTwoDatasetsCheck()
    suite = Suite('first suite', first_check, second_check)
    assert_that(len(suite.checks), equal_to(2))
    assert_that(suite[1], is_(second_check))
    assert_that(suite[0], is_(first_check))
    suite.remove(0)
    assert_that(calling(suite.__getitem__).with_args(0), raises(DeepchecksValueError, 'No index 0 in suite'))

def test_try_remove_unexisting_check_from_the_suite():
    if False:
        return 10
    suite = Suite('first suite', SimpleDatasetCheck(), SimpleTwoDatasetsCheck())
    assert_that(len(suite.checks), equal_to(2))
    assert_that(calling(suite.remove).with_args(3), raises(DeepchecksValueError, 'No index 3 in suite'))

def test_check_suite_instantiation_by_extending_another_check_suite():
    if False:
        for i in range(10):
            print('nop')
    suite = Suite('outer', tabular_checks.IsSingleValue(), Suite('inner1', tabular_checks.MixedNulls(), Suite('inner2', tabular_checks.MixedDataTypes()), tabular_checks.TrainTestPerformance()))
    assert all((not isinstance(c, Suite) for c in suite.checks))
    checks_types = [type(c) for c in suite.checks.values()]
    assert checks_types == [tabular_checks.IsSingleValue, tabular_checks.MixedNulls, tabular_checks.MixedDataTypes, tabular_checks.TrainTestPerformance]

def test_suite_result_checks_not_passed():
    if False:
        i = 10
        return i + 15
    result1 = CheckResult(0, 'check1')
    result1.conditions_results = [ConditionResult(ConditionCategory.PASS)]
    result2 = CheckResult(0, 'check2')
    result2.conditions_results = [ConditionResult(ConditionCategory.WARN)]
    result3 = CheckResult(0, 'check3')
    result3.conditions_results = [ConditionResult(ConditionCategory.FAIL)]
    not_passed_checks = SuiteResult('test', [result1, result2]).get_not_passed_checks()
    assert_that(not_passed_checks, has_length(1))
    not_passed_checks = SuiteResult('test', [result1, result2]).get_not_passed_checks(fail_if_warning=False)
    assert_that(not_passed_checks, has_length(0))
    not_passed_checks = SuiteResult('test', [result1, result2, result3]).get_not_passed_checks()
    assert_that(not_passed_checks, has_length(2))

def test_suite_result_passed_fn():
    if False:
        return 10
    result1 = CheckResult(0, 'check1')
    result1.conditions_results = [ConditionResult(ConditionCategory.PASS)]
    result2 = CheckResult(0, 'check2')
    result2.conditions_results = [ConditionResult(ConditionCategory.WARN)]
    result3 = CheckResult(0, 'check3')
    result3.conditions_results = [ConditionResult(ConditionCategory.FAIL)]
    result4 = CheckFailure(tabular_checks.IsSingleValue(), DeepchecksValueError(''))
    passed = SuiteResult('test', [result1, result2]).passed()
    assert_that(passed, equal_to(False))
    passed = SuiteResult('test', [result1, result2]).passed(fail_if_warning=False)
    assert_that(passed, equal_to(True))
    passed = SuiteResult('test', [result1, result2, result3]).passed(fail_if_warning=False)
    assert_that(passed, equal_to(False))
    passed = SuiteResult('test', [result1, result4]).passed()
    assert_that(passed, equal_to(True))
    passed = SuiteResult('test', [result1, result4]).passed(fail_if_check_not_run=True)
    assert_that(passed, equal_to(False))

def test_config():
    if False:
        i = 10
        return i + 15
    model_eval_suite = model_evaluation()
    check_amount = len(model_eval_suite.checks)
    suite_mod = model_eval_suite.config()
    assert_that(suite_mod, all_of(has_entry('module_name', 'deepchecks.tabular.suite'), has_entry('class_name', 'Suite'), has_entry('name', 'Model Evaluation Suite'), has_entry('version', __version__), has_entry('checks', instance_of(list))))
    conf_suite_mod = BaseSuite.from_config(suite_mod)
    assert_that(conf_suite_mod.name, equal_to('Model Evaluation Suite'))
    assert_that(conf_suite_mod.checks.values(), has_length(check_amount))