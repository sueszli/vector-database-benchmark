import sys
import math
sys.path.insert(1, '../../../')
from tests import pyunit_utils
from tests.pyunit_utils import dataset_prostate
from h2o.estimators.xgboost import H2OXGBoostEstimator

def test_eval_metric():
    if False:
        for i in range(10):
            print('nop')
    (train, _, _) = dataset_prostate()
    model = H2OXGBoostEstimator(ntrees=10, max_depth=4, score_each_iteration=True, seed=123)
    model.train(y='CAPSULE', x=train.names, training_frame=train)
    threshold = model._model_json['output']['default_threshold']
    scale = 100000.0
    xgb_threshold = math.floor(threshold * scale) / scale
    eval_metric = 'error@%s' % xgb_threshold
    print('Eval metric = ' + eval_metric)
    model_eval = H2OXGBoostEstimator(ntrees=10, max_depth=4, score_each_iteration=True, eval_metric=eval_metric, seed=123)
    model_eval.train(y='CAPSULE', x=train.names, training_frame=train)
    print(model_eval.scoring_history())
    h2o_error = model.scoring_history()['training_classification_error']
    h2o_error_last = h2o_error.iat[-1]
    xgb_error = model_eval.scoring_history()['training_custom']
    xgb_error_last = xgb_error.iat[-1]
    assert abs(h2o_error_last - xgb_error_last) < 1e-05
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_eval_metric)
else:
    test_eval_metric()