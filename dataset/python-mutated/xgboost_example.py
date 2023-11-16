from typing import Dict, List
import sklearn.datasets
import sklearn.metrics
import os
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.xgboost import TuneReportCheckpointCallback

def train_breast_cancer(config: dict):
    if False:
        i = 10
        return i + 15
    (data, labels) = sklearn.datasets.load_breast_cancer(return_X_y=True)
    (train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.25)
    train_set = xgb.DMatrix(train_x, label=train_y)
    test_set = xgb.DMatrix(test_x, label=test_y)
    xgb.train(config, train_set, evals=[(test_set, 'test')], verbose_eval=False, callbacks=[TuneReportCheckpointCallback(filename='model.xgb', frequency=1)])

def train_breast_cancer_cv(config: dict):
    if False:
        while True:
            i = 10
    (data, labels) = sklearn.datasets.load_breast_cancer(return_X_y=True)

    def average_cv_folds(results_dict: Dict[str, List[float]]) -> Dict[str, float]:
        if False:
            i = 10
            return i + 15
        return {k: np.mean(v) for (k, v) in results_dict.items()}
    train_set = xgb.DMatrix(data, label=labels)
    xgb.cv(config, train_set, verbose_eval=False, stratified=True, callbacks=[TuneReportCheckpointCallback(results_postprocessing_fn=average_cv_folds, frequency=0)])

def get_best_model_checkpoint(best_result: 'ray.train.Result'):
    if False:
        i = 10
        return i + 15
    best_bst = xgb.Booster()
    with best_result.checkpoint.as_directory() as checkpoint_dir:
        best_bst.load_model(os.path.join(checkpoint_dir, 'model.xgb'))
    accuracy = 1.0 - best_result.metrics['test-error']
    print(f'Best model parameters: {best_result.config}')
    print(f'Best model total accuracy: {accuracy:.4f}')
    return best_bst

def tune_xgboost(use_cv: bool=False):
    if False:
        return 10
    search_space = {'objective': 'binary:logistic', 'eval_metric': ['logloss', 'error'], 'max_depth': tune.randint(1, 9), 'min_child_weight': tune.choice([1, 2, 3]), 'subsample': tune.uniform(0.5, 1.0), 'eta': tune.loguniform(0.0001, 0.1)}
    scheduler = ASHAScheduler(max_t=10, grace_period=1, reduction_factor=2)
    tuner = tune.Tuner(tune.with_resources(train_breast_cancer if not use_cv else train_breast_cancer_cv, resources={'cpu': 1}), tune_config=tune.TuneConfig(metric='test-logloss', mode='min', num_samples=10, scheduler=scheduler), param_space=search_space)
    results = tuner.fit()
    return results.get_best_result()
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cv', action='store_true', help='Use `xgb.cv` instead of `xgb.train`.')
    (args, _) = parser.parse_known_args()
    best_result = tune_xgboost(args.use_cv)
    if not args.use_cv:
        best_bst = get_best_model_checkpoint(best_result)