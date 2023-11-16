"""Tests for Model Info."""
from hamcrest import assert_that, calling, has_entries, raises
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from deepchecks.core.errors import ModelValidationError
from deepchecks.tabular.checks.model_evaluation.model_info import ModelInfo

def assert_model_result(result):
    if False:
        print('Hello World!')
    assert_that(result.value, has_entries(type='AdaBoostClassifier', params=has_entries(algorithm='SAMME.R', learning_rate=1, n_estimators=50)))

def test_model_info_function(iris_adaboost):
    if False:
        print('Hello World!')
    result = ModelInfo().run(iris_adaboost)
    assert_model_result(result)

def test_model_info_object(iris_adaboost):
    if False:
        return 10
    mi = ModelInfo()
    result = mi.run(iris_adaboost)
    assert_model_result(result)

def test_model_info_pipeline(iris_adaboost):
    if False:
        print('Hello World!')
    simple_pipeline = Pipeline([('nan_handling', SimpleImputer(strategy='most_frequent')), ('adaboost', iris_adaboost)])
    result = ModelInfo().run(simple_pipeline)
    assert_model_result(result)

def test_model_info_wrong_input():
    if False:
        print('Hello World!')
    assert_that(calling(ModelInfo().run).with_args('some string'), raises(ModelValidationError, 'Model supplied does not meets the minimal interface requirements. Read more about .*'))