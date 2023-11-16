"""Unused features tests."""
from hamcrest import assert_that, equal_to, greater_than, has_length
from deepchecks.tabular.checks.model_evaluation import UnusedFeatures

def test_unused_feature_detection(iris_split_dataset_and_model_rf):
    if False:
        return 10
    (_, test_ds, clf) = iris_split_dataset_and_model_rf
    result = UnusedFeatures().run(test_ds, clf)
    assert_that(set(result.value['used features']), equal_to({'petal width (cm)', 'petal length (cm)', 'sepal length (cm)'}))
    assert_that(set(result.value['unused features']['high variance']), equal_to({'sepal width (cm)'}))
    assert_that(set(result.value['unused features']['low variance']), equal_to(set()))
    assert_that(result.display, has_length(greater_than(0)))

def test_unused_feature_detection_without_display(iris_split_dataset_and_model_rf):
    if False:
        i = 10
        return i + 15
    (_, test_ds, clf) = iris_split_dataset_and_model_rf
    result = UnusedFeatures().run(test_ds, clf, with_display=False)
    assert_that(set(result.value['used features']), equal_to({'petal width (cm)', 'petal length (cm)', 'sepal length (cm)'}))
    assert_that(set(result.value['unused features']['high variance']), equal_to({'sepal width (cm)'}))
    assert_that(set(result.value['unused features']['low variance']), equal_to(set()))
    assert_that(result.display, has_length(0))

def test_low_feature_importance_threshold(iris_split_dataset_and_model_rf):
    if False:
        while True:
            i = 10
    (_, test_ds, clf) = iris_split_dataset_and_model_rf
    result = UnusedFeatures(feature_importance_threshold=0).run(test_ds, clf)
    assert_that(set(result.value['used features']), equal_to({'petal width (cm)', 'petal length (cm)', 'sepal width (cm)', 'sepal length (cm)'}))
    assert_that(set(result.value['unused features']['high variance']), equal_to(set()))
    assert_that(set(result.value['unused features']['low variance']), equal_to(set()))

def test_higher_variance_threshold(iris_split_dataset_and_model_rf):
    if False:
        return 10
    (_, test_ds, clf) = iris_split_dataset_and_model_rf
    result = UnusedFeatures(feature_variance_threshold=2).run(test_ds, clf)
    assert_that(set(result.value['used features']), equal_to({'petal width (cm)', 'petal length (cm)', 'sepal length (cm)'}))
    assert_that(set(result.value['unused features']['high variance']), equal_to(set()))
    assert_that(set(result.value['unused features']['low variance']), equal_to({'sepal width (cm)'}))