"""Test user utils"""
from hamcrest import assert_that, calling, close_to, raises
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular.feature_importance import calculate_feature_importance
from deepchecks.tabular.metric_utils import DeepcheckScorer

def test_calculate_importance(iris_split_dataset_and_model):
    if False:
        for i in range(10):
            print('nop')
    (train_ds, _, adaboost) = iris_split_dataset_and_model
    fi = calculate_feature_importance(adaboost, train_ds)
    assert_that(fi.sum(), close_to(1, 1e-06))

def test_calculate_importance_with_kwargs(iris_split_dataset_and_model):
    if False:
        i = 10
        return i + 15
    (train_ds, _, adaboost) = iris_split_dataset_and_model
    scorer = DeepcheckScorer('accuracy', [0, 1, 2], [0, 1, 2])
    fi = calculate_feature_importance(adaboost, train_ds, n_repeats=30, mask_high_variance_features=False, n_samples=10000, alternative_scorer=scorer)
    assert_that(fi.sum(), close_to(1, 1e-06))

def test_calculate_importance_force_permutation_fail_on_dataframe(iris_split_dataset_and_model):
    if False:
        while True:
            i = 10
    (train_ds, _, adaboost) = iris_split_dataset_and_model
    df_only_features = train_ds.data.drop(train_ds.label_name, axis=1)
    assert_that(calling(calculate_feature_importance).with_args(adaboost, df_only_features), raises(DeepchecksValueError, 'Cannot calculate permutation feature importance on a pandas Dataframe'))