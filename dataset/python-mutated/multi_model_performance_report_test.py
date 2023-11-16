import pytest
from hamcrest import assert_that, has_length
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from deepchecks.tabular.checks.model_evaluation import MultiModelPerformanceReport

@pytest.fixture
def classification_models(iris_split_dataset_and_model):
    if False:
        print('Hello World!')
    (train, test, model) = iris_split_dataset_and_model
    model2 = RandomForestClassifier(random_state=0)
    model2.fit(train.data[train.features], train.data[train.label_name])
    model3 = DecisionTreeClassifier(random_state=0)
    model3.fit(train.data[train.features], train.data[train.label_name])
    return (train, test, model, model2, model3)

@pytest.fixture
def regression_models(diabetes_split_dataset_and_model):
    if False:
        return 10
    (train, test, model) = diabetes_split_dataset_and_model
    model2 = RandomForestRegressor(random_state=0)
    model2.fit(train.data[train.features], train.data[train.label_name])
    model3 = DecisionTreeRegressor(random_state=0)
    model3.fit(train.data[train.features], train.data[train.label_name])
    return (train, test, model, model2, model3)

def test_multi_classification(classification_models):
    if False:
        for i in range(10):
            print('nop')
    (train, test, model, model2, model3) = classification_models
    result = MultiModelPerformanceReport().run(train, test, [model, model2, model3])
    assert_that(result.value, has_length(27))

def test_regression(regression_models):
    if False:
        print('Hello World!')
    (train, test, model, model2, model3) = regression_models
    result = MultiModelPerformanceReport().run(train, test, [model, model2, model3])
    assert_that(result.value, has_length(9))