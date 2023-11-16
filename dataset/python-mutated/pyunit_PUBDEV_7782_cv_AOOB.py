import sys, os
sys.path.insert(1, os.path.join('..', '..', '..'))
import h2o
import h2o.exceptions
from tests import pyunit_utils
from h2o.estimators import H2OGeneralizedLinearEstimator

def test_GLM_throws_ArrayOutOfBoundException():
    if False:
        return 10
    df = h2o.import_file(path=pyunit_utils.locate('smalldata/prostate/prostate.csv'))
    target = 'CAPSULE'
    nFold = 5
    for col in [target, 'GLEASON']:
        df[col] = df[col].asfactor()
        glm = H2OGeneralizedLinearEstimator(lambda_search=True, alpha=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], nfolds=nFold, seed=12345)
        glm.train(y=target, training_frame=df)
        assert len(glm._model_json['output']['cross_validation_models']) == nFold, 'expected number of cross_validation_model: {0}.  Actual number of cross_validation: {1}'.format(len(glm._model_json['output']['cross_validation_models']), nFold)
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_GLM_throws_ArrayOutOfBoundException)
else:
    test_GLM_throws_ArrayOutOfBoundException()