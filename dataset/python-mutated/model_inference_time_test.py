import re
import typing as t
from hamcrest import assert_that, instance_of, matches_regexp, only_contains
from deepchecks.core import CheckResult, ConditionCategory
from deepchecks.tabular.checks.model_evaluation import ModelInferenceTime
from deepchecks.tabular.dataset import Dataset
from tests.base.utils import SCIENTIFIC_NOTATION_REGEXP, equal_condition_result

def test_model_inference_time_check(iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, object]):
    if False:
        return 10
    (_, test, model) = iris_split_dataset_and_model
    check = ModelInferenceTime()
    result = check.run(test, model)
    assert_that(result, instance_of(CheckResult))
    assert_that(result.value, instance_of(float))
    assert_that(result.display, instance_of(list))
    assert_that(result.display, only_contains(instance_of(str)))
    details_pattern = f'Average model inference time for one sample \\(in seconds\\): {SCIENTIFIC_NOTATION_REGEXP.pattern}'
    assert_that(result.display[0], matches_regexp(details_pattern))

def test_model_inference_time_check_with_condition_that_should_pass(iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, object]):
    if False:
        i = 10
        return i + 15
    (_, test, model) = iris_split_dataset_and_model
    check = ModelInferenceTime().add_condition_inference_time_less_than(0.1)
    result = check.run(test, model)
    (condition_result, *_) = check.conditions_decision(result)
    name = 'Average model inference time for one sample is less than 0.1'
    details_pattern = re.compile(f'Found average inference time \\(seconds\\): {SCIENTIFIC_NOTATION_REGEXP.pattern}')
    assert_that(condition_result, equal_condition_result(is_pass=True, category=ConditionCategory.PASS, name=name, details=details_pattern))

def test_model_inference_time_check_with_condition_that_should_not_pass(iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, object]):
    if False:
        for i in range(10):
            print('nop')
    (_, test, model) = iris_split_dataset_and_model
    check = ModelInferenceTime().add_condition_inference_time_less_than(1e-08)
    result = check.run(test, model)
    (condition_result, *_) = check.conditions_decision(result)
    name = 'Average model inference time for one sample is less than 1e-08'
    details_pattern = re.compile(f'Found average inference time \\(seconds\\): {SCIENTIFIC_NOTATION_REGEXP.pattern}')
    assert_that(condition_result, equal_condition_result(is_pass=False, category=ConditionCategory.FAIL, name=name, details=details_pattern))