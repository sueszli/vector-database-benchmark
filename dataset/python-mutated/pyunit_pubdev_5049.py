from builtins import range
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.gbm import H2OGradientBoostingEstimator

def get_ntrees(model):
    if False:
        i = 10
        return i + 15
    return max(model._model_json['output']['scoring_history']['number_of_trees'])

def demo_xval_with_validation_frame():
    if False:
        for i in range(10):
            print('nop')
    prostate = h2o.import_file(path=pyunit_utils.locate('smalldata/logreg/prostate.csv'))
    prostate[1] = prostate[1].asfactor()
    print(prostate.summary())
    prostate_inverse = h2o.import_file(path=pyunit_utils.locate('smalldata/logreg/prostate.csv'))
    resp = 1 - prostate_inverse[1]
    prostate_inverse[1] = resp.asfactor()
    print(prostate_inverse.summary())
    ntrees = 50
    X = list(range(2, 9))
    y = 1
    prostate_gbm = H2OGradientBoostingEstimator(nfolds=5, ntrees=ntrees, distribution='bernoulli', seed=1, score_each_iteration=True, stopping_rounds=3)
    prostate_gbm.train(x=X, y=y, training_frame=prostate)
    prostate_gbm.show()
    assert get_ntrees(prostate_gbm) < ntrees
    prostate_gbm_noxval = H2OGradientBoostingEstimator(ntrees=ntrees, distribution='bernoulli', seed=1, score_each_iteration=True, stopping_rounds=3)
    prostate_gbm_noxval.train(x=X, y=y, training_frame=prostate, validation_frame=prostate_inverse)
    prostate_gbm_noxval.show()
    assert get_ntrees(prostate_gbm_noxval) == 6
    assert get_ntrees(prostate_gbm_noxval) < get_ntrees(prostate_gbm)
    prostate_gbm_v = H2OGradientBoostingEstimator(nfolds=5, ntrees=ntrees, distribution='bernoulli', seed=1, score_each_iteration=True, stopping_rounds=3)
    prostate_gbm_v.train(x=X, y=y, training_frame=prostate, validation_frame=prostate_inverse)
    prostate_gbm_v.show()
    pyunit_utils.check_models(prostate_gbm, prostate_gbm_v)
    assert get_ntrees(prostate_gbm) == get_ntrees(prostate_gbm_v)
if __name__ == '__main__':
    pyunit_utils.standalone_test(demo_xval_with_validation_frame)
else:
    demo_xval_with_validation_frame()