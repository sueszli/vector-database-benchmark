"""Tests for segment performance check."""
from hamcrest import assert_that, calling, close_to, equal_to, greater_than, has_entries, has_length, has_property, raises
from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError
from deepchecks.tabular.checks.model_evaluation.segment_performance import SegmentPerformance
from deepchecks.tabular.dataset import Dataset

def test_dataset_wrong_input():
    if False:
        print('Hello World!')
    bad_dataset = 'wrong_input'
    assert_that(calling(SegmentPerformance().run).with_args(bad_dataset, None), raises(DeepchecksValueError, 'non-empty instance of Dataset or DataFrame was expected, instead got str'))

def test_dataset_no_label(iris, iris_adaboost):
    if False:
        for i in range(10):
            print('nop')
    iris = iris.drop('target', axis=1)
    iris_dataset = Dataset(iris)
    assert_that(calling(SegmentPerformance().run).with_args(iris_dataset, iris_adaboost), raises(DeepchecksNotSupportedError, 'Dataset does not contain a label column'))

def test_segment_performance_diabetes(diabetes_split_dataset_and_model):
    if False:
        return 10
    (_, val, model) = diabetes_split_dataset_and_model
    result = SegmentPerformance(feature_1='age', feature_2='sex').run(val, model)
    assert_that(result.value, has_entries({'scores': has_property('shape', (10, 2)), 'counts': has_property('shape', (10, 2))}))
    assert_that(result.value['scores'].mean(), close_to(-53, 1))
    assert_that(result.value['counts'].sum(), equal_to(146))
    assert_that(result.display, has_length(greater_than(0)))

def test_segment_performance_diabetes_without_display(diabetes_split_dataset_and_model):
    if False:
        i = 10
        return i + 15
    (_, val, model) = diabetes_split_dataset_and_model
    result = SegmentPerformance(feature_1='age', feature_2='sex').run(val, model, with_display=False)
    assert_that(result.value, has_entries({'scores': has_property('shape', (10, 2)), 'counts': has_property('shape', (10, 2))}))
    assert_that(result.value['scores'].mean(), close_to(-53, 1))
    assert_that(result.value['counts'].sum(), equal_to(146))
    assert_that(result.display, has_length(0))

def test_segment_performance_illegal_features(diabetes_split_dataset_and_model):
    if False:
        for i in range(10):
            print('nop')
    (_, val, model) = diabetes_split_dataset_and_model
    assert_that(calling(SegmentPerformance(feature_1='AGE', feature_2='sex').run).with_args(val, model), raises(DeepchecksValueError, '\\"feature_1\\" and \\"feature_2\\" must be in dataset columns'))

def test_segment_performance_non_cat_or_num(kiss_dataset_and_model):
    if False:
        while True:
            i = 10
    (_, val, model) = kiss_dataset_and_model
    assert_that(calling(SegmentPerformance(feature_1='numeric_label', feature_2='binary_feature').run).with_args(val, model), raises(DeepchecksValueError, '\\"feature_1\\" must be numerical or categorical, but it neither.'))

def test_segment_top_features(diabetes_split_dataset_and_model):
    if False:
        while True:
            i = 10
    (_, val, model) = diabetes_split_dataset_and_model
    result = SegmentPerformance().run(val, model).value
    assert_that(result, has_entries({'scores': has_property('shape', (10, 10)), 'counts': has_property('shape', (10, 10))}))
    assert_that(result['counts'].sum(), equal_to(146))