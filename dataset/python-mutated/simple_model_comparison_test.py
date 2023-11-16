"""Contains unit tests for the confusion_matrix_report check."""
from hamcrest import assert_that, calling, close_to, equal_to, greater_than, has_entries, has_entry, has_items, has_length, is_, raises
from sklearn.metrics import f1_score, get_scorer, make_scorer, recall_score
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular.checks.model_evaluation import SimpleModelComparison
from deepchecks.tabular.metric_utils.scorers import get_default_scorers
from deepchecks.tabular.utils.task_type import TaskType
from tests.base.utils import equal_condition_result

def test_dataset_wrong_input():
    if False:
        print('Hello World!')
    bad_dataset = 'wrong_input'
    assert_that(calling(SimpleModelComparison().run).with_args(bad_dataset, bad_dataset, None), raises(DeepchecksValueError, 'non-empty instance of Dataset or DataFrame was expected, instead got str'))

def test_classification_random(iris_split_dataset_and_model):
    if False:
        print('Hello World!')
    (train_ds, test_ds, clf) = iris_split_dataset_and_model
    check = SimpleModelComparison(strategy='stratified')
    result = check.run(train_ds, test_ds, clf).value
    assert_classification(result, [0, 1, 2])

def test_classification_uniform(iris_split_dataset_and_model):
    if False:
        i = 10
        return i + 15
    (train_ds, test_ds, clf) = iris_split_dataset_and_model
    check = SimpleModelComparison(strategy='uniform')
    result = check.run(train_ds, test_ds, clf).value
    assert_classification(result, [0, 1, 2])

def test_classification_constant(iris_split_dataset_and_model):
    if False:
        print('Hello World!')
    (train_ds, test_ds, clf) = iris_split_dataset_and_model
    check = SimpleModelComparison(strategy='most_frequent')
    result = check.run(train_ds, test_ds, clf).value
    assert_classification(result, [0, 1, 2])

def test_classification_binary_string_labels(iris_binary_string_split_dataset_and_model):
    if False:
        return 10
    (train_ds, test_ds, clf) = iris_binary_string_split_dataset_and_model
    check = SimpleModelComparison()
    result = check.run(train_ds, test_ds, clf).value
    assert_classification(result, ['a', 'b'])

def test_classification_binary_string_labels_custom_scorer(iris_binary_string_split_dataset_and_model):
    if False:
        while True:
            i = 10
    (train_ds, test_ds, clf) = iris_binary_string_split_dataset_and_model
    check = SimpleModelComparison(scorers=[get_scorer('f1'), make_scorer(recall_score, average=None, zero_division=0)])
    result = check.run(train_ds, test_ds, clf).value
    assert_that(result, equal_to({'scores': {'f1_score': {'Origin': 0.9411764705882353, 'Simple': 0.0}, 'recall_score': {'a': {'Origin': 0.9411764705882353, 'Simple': 1.0}, 'b': {'Origin': 1.0, 'Simple': 0.0}}}, 'type': TaskType.BINARY, 'scorers_perfect': {'f1_score': 1.0, 'recall_score': 1.0}}))

def test_classification_random_custom_metric(iris_split_dataset_and_model):
    if False:
        for i in range(10):
            print('nop')
    (train_ds, test_ds, clf) = iris_split_dataset_and_model
    check = SimpleModelComparison(strategy='stratified', scorers={'recall': make_scorer(recall_score, average=None)})
    result = check.run(train_ds, test_ds, clf)
    assert_classification(result.value, [0, 1, 2], ['recall'])
    assert_that(result.display, has_length(greater_than(0)))

def test_classification_random_custom_metric_without_display(iris_split_dataset_and_model):
    if False:
        for i in range(10):
            print('nop')
    (train_ds, test_ds, clf) = iris_split_dataset_and_model
    check = SimpleModelComparison(strategy='stratified', scorers={'recall': make_scorer(recall_score, average=None)})
    result = check.run(train_ds, test_ds, clf, with_display=False)
    assert_classification(result.value, [0, 1, 2], ['recall'])
    assert_that(result.display, has_length(0))

def test_regression_random(diabetes_split_dataset_and_model):
    if False:
        for i in range(10):
            print('nop')
    (train_ds, test_ds, clf) = diabetes_split_dataset_and_model
    check = SimpleModelComparison(strategy='stratified')
    result = check.run(train_ds, test_ds, clf).value
    assert_regression(result)

def test_regression_random_state(diabetes_split_dataset_and_model):
    if False:
        print('Hello World!')
    (train_ds, test_ds, clf) = diabetes_split_dataset_and_model
    check = SimpleModelComparison(strategy='stratified', random_state=0)
    result = check.run(train_ds, test_ds, clf).value
    assert_regression(result)

def test_regression_constant(diabetes_split_dataset_and_model):
    if False:
        for i in range(10):
            print('nop')
    (train_ds, test_ds, clf) = diabetes_split_dataset_and_model
    check = SimpleModelComparison(strategy='most_frequent')
    result = check.run(train_ds, test_ds, clf).value
    assert_regression(result)

def test_regression_uniform(diabetes_split_dataset_and_model):
    if False:
        print('Hello World!')
    (train_ds, test_ds, clf) = diabetes_split_dataset_and_model
    check = SimpleModelComparison(strategy='uniform')
    result = check.run(train_ds, test_ds, clf).value
    assert_regression(result)

def test_condition_ratio_not_less_than_not_passed(diabetes_split_dataset_and_model):
    if False:
        i = 10
        return i + 15
    (train_ds, test_ds, clf) = diabetes_split_dataset_and_model
    check = SimpleModelComparison().add_condition_gain_greater_than(0.4)
    check_result = check.run(train_ds, test_ds, clf)
    condition_result = check_result.conditions_results
    assert_that(condition_result, has_items(equal_condition_result(is_pass=False, name='Model performance gain over simple model is greater than 40%', details="Found failed metrics: {'Neg RMSE': '24.32%'}")))

def test_condition_failed_for_multiclass(iris_split_dataset_and_model):
    if False:
        i = 10
        return i + 15
    (train_ds, test_ds, clf) = iris_split_dataset_and_model
    check = SimpleModelComparison(strategy='most_frequent').add_condition_gain_greater_than(0.8)
    result = check.run(train_ds, test_ds, clf)
    assert_that(result.conditions_results, has_items(equal_condition_result(is_pass=False, name='Model performance gain over simple model is greater than 80%', details="Found classes with failed metric's gain: {1: {'F1': '78.15%'}}")))

def test_condition_pass_for_multiclass_avg(iris_split_dataset_and_model):
    if False:
        return 10
    (train_ds, test_ds, clf) = iris_split_dataset_and_model
    check = SimpleModelComparison(strategy='most_frequent').add_condition_gain_greater_than(0.43, average=True)
    result = check.run(train_ds, test_ds, clf)
    assert_that(result.conditions_results, has_items(equal_condition_result(is_pass=True, details="All metrics passed, metric's gain: {'F1': '89.74%'}", name='Model performance gain over simple model is greater than 43%')))

def test_condition_pass_for_custom_scorer(iris_dataset_single_class, iris_random_forest_single_class):
    if False:
        for i in range(10):
            print('nop')
    train_ds = iris_dataset_single_class
    test_ds = iris_dataset_single_class
    clf = iris_random_forest_single_class
    check = SimpleModelComparison(scorers=['f1'], strategy='most_frequent').add_condition_gain_greater_than(0.43)
    result = check.run(train_ds, test_ds, clf)
    assert_that(result.conditions_results, has_items(equal_condition_result(is_pass=True, details="Found metrics with perfect score, no gain is calculated: ['f1']", name='Model performance gain over simple model is greater than 43%')))

def test_condition_pass_for_multiclass_avg_with_classes(iris_split_dataset_and_model):
    if False:
        for i in range(10):
            print('nop')
    (train_ds, test_ds, clf) = iris_split_dataset_and_model
    check = SimpleModelComparison(strategy='most_frequent').add_condition_gain_greater_than(1, average=False).add_condition_gain_greater_than(1, average=True, classes=[0])
    result = check.run(train_ds, test_ds, clf)
    assert_that(result.conditions_results, has_items(equal_condition_result(is_pass=False, name='Model performance gain over simple model is greater than 100%', details="Found classes with failed metric's gain: {1: {'F1': '78.15%'}, 2: {'F1': '85.71%'}}"), equal_condition_result(is_pass=True, details="Found metrics with perfect score, no gain is calculated: ['F1']", name='Model performance gain over simple model is greater than 100% for classes [0]')))

def test_condition_pass_for_new_test_classes(kiss_dataset_and_model):
    if False:
        while True:
            i = 10
    (train_ds, test_ds, clf) = kiss_dataset_and_model
    check = SimpleModelComparison(strategy='most_frequent').add_condition_gain_greater_than(1)
    result = check.run(train_ds, test_ds, clf)
    assert_that(result.conditions_results, has_items(equal_condition_result(is_pass=True, details="Found metrics with perfect score, no gain is calculated: ['F1']", name='Model performance gain over simple model is greater than 100%')))

def test_condition_ratio_not_less_than_passed(diabetes_split_dataset_and_model):
    if False:
        return 10
    (train_ds, test_ds, clf) = diabetes_split_dataset_and_model
    check = SimpleModelComparison(strategy='stratified', n_samples=None).add_condition_gain_greater_than()
    check_result = check.run(train_ds, test_ds, clf)
    condition_result = check_result.conditions_results
    assert_that(condition_result, has_items(equal_condition_result(is_pass=True, details="All metrics passed, metric's gain: {'Neg RMSE': '49.7%'}", name='Model performance gain over simple model is greater than 10%')))

def test_classification_tree(iris_split_dataset_and_model):
    if False:
        while True:
            i = 10
    (train_ds, test_ds, clf) = iris_split_dataset_and_model
    check = SimpleModelComparison(strategy='tree')
    result = check.run(train_ds, test_ds, clf).value
    assert_classification(result, [0, 1, 2])

def test_classification_tree_custom_metric(iris_split_dataset_and_model):
    if False:
        return 10
    (train_ds, test_ds, clf) = iris_split_dataset_and_model
    check = SimpleModelComparison(strategy='tree', scorers={'recall': make_scorer(recall_score, average=None), 'f1': make_scorer(f1_score, average=None)})
    result = check.run(train_ds, test_ds, clf).value
    assert_classification(result, [0, 1, 2], ['recall', 'f1'])

def test_regression_constant(diabetes_split_dataset_and_model):
    if False:
        while True:
            i = 10
    (train_ds, test_ds, clf) = diabetes_split_dataset_and_model
    check = SimpleModelComparison(strategy='most_frequent')
    result = check.run(train_ds, test_ds, clf).value
    assert_regression(result)

def test_regression_tree(diabetes_split_dataset_and_model):
    if False:
        while True:
            i = 10
    (train_ds, test_ds, clf) = diabetes_split_dataset_and_model
    check = SimpleModelComparison(strategy='tree')
    result = check.run(train_ds, test_ds, clf).value
    assert_regression(result)

def test_regression_tree_random_state(diabetes_split_dataset_and_model):
    if False:
        return 10
    (train_ds, test_ds, clf) = diabetes_split_dataset_and_model
    check = SimpleModelComparison(strategy='tree', random_state=55)
    result = check.run(train_ds, test_ds, clf).value
    assert_regression(result)

def test_regression_tree_max_depth(diabetes_split_dataset_and_model):
    if False:
        print('Hello World!')
    (train_ds, test_ds, clf) = diabetes_split_dataset_and_model
    check = SimpleModelComparison(strategy='tree', max_depth=5)
    result = check.run(train_ds, test_ds, clf).value
    assert_regression(result)

def assert_regression(result):
    if False:
        i = 10
        return i + 15
    default_scorers = get_default_scorers(TaskType.REGRESSION)
    metric = next(iter(default_scorers))
    assert_that(result['scores'], has_entry(metric, has_entries({'Origin': close_to(-100, 100), 'Simple': close_to(-100, 100)})))
    assert_that(result['scorers_perfect'], has_entry(metric, is_(0)))

def assert_classification(result, classes, metrics=None):
    if False:
        for i in range(10):
            print('nop')
    if not metrics:
        default_scorers = get_default_scorers(TaskType.MULTICLASS, class_avg=False)
        metrics = [next(iter(default_scorers))]
    class_matchers = {clas: has_entries({'Origin': close_to(1, 1), 'Simple': close_to(1, 1)}) for clas in classes}
    matchers = {metric: has_entries(class_matchers) for metric in metrics}
    assert_that(result['scores'], has_entries(matchers))
    assert_that(result['scorers_perfect'], has_entries({metric: is_(1) for metric in metrics}))