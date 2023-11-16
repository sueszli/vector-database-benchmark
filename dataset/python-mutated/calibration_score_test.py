"""Contains unit tests for the calibration_metric check."""
from hamcrest import assert_that, calling, close_to, greater_than, has_entries, has_length, raises
from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError, ModelValidationError
from deepchecks.tabular.checks.model_evaluation import CalibrationScore
from deepchecks.tabular.dataset import Dataset

def test_dataset_wrong_input():
    if False:
        print('Hello World!')
    bad_dataset = 'wrong_input'
    assert_that(calling(CalibrationScore().run).with_args(bad_dataset, None), raises(DeepchecksValueError, 'non-empty instance of Dataset or DataFrame was expected, instead got str'))

def test_dataset_no_label(iris, iris_adaboost):
    if False:
        print('Hello World!')
    iris = iris.drop('target', axis=1)
    ds = Dataset(iris)
    assert_that(calling(CalibrationScore().run).with_args(ds, iris_adaboost), raises(DeepchecksNotSupportedError, 'Dataset does not contain a label column'))

def test_regresion_model(diabetes_split_dataset_and_model):
    if False:
        return 10
    (train, _, clf) = diabetes_split_dataset_and_model
    assert_that(calling(CalibrationScore().run).with_args(train, clf), raises(ModelValidationError, 'Check is irrelevant for regression tasks'))

def test_model_info_object(iris_labeled_dataset, iris_adaboost):
    if False:
        print('Hello World!')
    check = CalibrationScore()
    result = check.run(iris_labeled_dataset, iris_adaboost)
    assert_that(result.value, has_length(3))
    assert_that(result.value, has_entries({0: close_to(0.0, 0.0001), 1: close_to(0.026, 0.001), 2: close_to(0.026, 0.001)}))
    assert_that(result.display, has_length(greater_than(0)))

def test_model_info_object_without_display(iris_labeled_dataset, iris_adaboost):
    if False:
        i = 10
        return i + 15
    check = CalibrationScore()
    result = check.run(iris_labeled_dataset, iris_adaboost, with_display=False)
    assert_that(result.value, has_length(3))
    assert_that(result.value, has_entries({0: close_to(0.0, 0.0001), 1: close_to(0.026, 0.001), 2: close_to(0.026, 0.001)}))
    assert_that(result.display, has_length(0))

def test_binary_model_info_object(iris_dataset_single_class_labeled, iris_random_forest_single_class):
    if False:
        return 10
    check = CalibrationScore()
    result = check.run(iris_dataset_single_class_labeled, iris_random_forest_single_class).value
    assert_that(result, has_length(1))
    assert_that(result, has_entries({0: close_to(0.0002, 0.0005)}))

def test_binary_string_model_info_object(iris_binary_string_split_dataset_and_model):
    if False:
        i = 10
        return i + 15
    (_, test_ds, clf) = iris_binary_string_split_dataset_and_model
    check = CalibrationScore()
    result = check.run(test_ds, clf).value
    assert_that(result, has_length(1))
    assert_that(result, has_entries({0: close_to(0.04, 0.001)}))