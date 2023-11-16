import sys, os
sys.path.insert(1, os.path.join('..', '..', '..'))
import h2o
import h2o.exceptions
from tests import pyunit_utils
from h2o.estimators import H2OGeneralizedLinearEstimator

def test_GLM_throws_ArrayOutOfBoundException():
    if False:
        print('Hello World!')
    nFold = 5
    fr = h2o.import_file(pyunit_utils.locate('bigdata/laptop/jira/christine.arff'))
    splitFrame = fr.split_frame(ratios=[0.05])
    glm = H2OGeneralizedLinearEstimator(family='binomial', nfolds=nFold, lambda_search=True, alpha=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    glm.train(y=0, training_frame=splitFrame[0])
    assert len(glm._model_json['output']['cross_validation_models']) == nFold, 'expected number of cross_validation_model: {0}.  Actual number of cross_validation: {1}'.format(len(glm._model_json['output']['cross_validation_models']), nFold)
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_GLM_throws_ArrayOutOfBoundException)
else:
    test_GLM_throws_ArrayOutOfBoundException()