"""Test task type inference"""
from hamcrest import assert_that, equal_to, has_items, is_
from deepchecks.tabular import Context
from deepchecks.tabular.utils.task_type import TaskType

def test_infer_task_type_binary(iris_dataset_single_class, iris_random_forest_single_class):
    if False:
        for i in range(10):
            print('nop')
    context = Context(iris_dataset_single_class, model=iris_random_forest_single_class)
    assert_that(context.task_type, equal_to(TaskType.BINARY))
    assert_that(context.observed_classes, has_items(0, 1))
    assert_that(context._model_classes, has_items(0, 1))

def test_infer_task_type_multiclass(iris_split_dataset_and_model_rf):
    if False:
        return 10
    (train_ds, _, clf) = iris_split_dataset_and_model_rf
    context = Context(train_ds, model=clf)
    assert_that(context.task_type, equal_to(TaskType.MULTICLASS))
    assert_that(context.observed_classes, has_items(0, 1, 2))
    assert_that(context._model_classes, has_items(0, 1, 2))

def test_infer_task_type_regression(diabetes, diabetes_model):
    if False:
        for i in range(10):
            print('nop')
    (train_ds, _) = diabetes
    context = Context(train_ds, model=diabetes_model)
    assert_that(context.task_type, equal_to(TaskType.REGRESSION))
    assert_that(context._model_classes, is_(None))

def test_task_type_not_sklearn_regression(diabetes):
    if False:
        i = 10
        return i + 15

    class RegressionModel:

        def predict(self, x):
            if False:
                for i in range(10):
                    print('nop')
            return [0] * len(x)
    (train_ds, _) = diabetes
    context = Context(train_ds, model=RegressionModel())
    assert_that(context.task_type, equal_to(TaskType.REGRESSION))
    assert_that(context._model_classes, is_(None))

def test_task_type_not_sklearn_binary(iris_dataset_single_class):
    if False:
        print('Hello World!')

    class ClassificationModel:

        def predict(self, x):
            if False:
                for i in range(10):
                    print('nop')
            return [0] * len(x)

        def predict_proba(self, x):
            if False:
                i = 10
                return i + 15
            return [[1, 0]] * len(x)
    context = Context(iris_dataset_single_class, model=ClassificationModel())
    assert_that(context.task_type, equal_to(TaskType.BINARY))
    assert_that(context.observed_classes, has_items(0, 1))
    assert_that(context._model_classes, is_(None))

def test_task_type_not_sklearn_multiclass(iris_labeled_dataset):
    if False:
        print('Hello World!')

    class ClassificationModel:

        def predict(self, x):
            if False:
                for i in range(10):
                    print('nop')
            return [0] * len(x)

        def predict_proba(self, x):
            if False:
                for i in range(10):
                    print('nop')
            return [[1, 0]] * len(x)
    context = Context(iris_labeled_dataset, model=ClassificationModel())
    assert_that(context.task_type, equal_to(TaskType.MULTICLASS))
    assert_that(context.observed_classes, has_items(0, 1, 2))
    assert_that(context._model_classes, is_(None))