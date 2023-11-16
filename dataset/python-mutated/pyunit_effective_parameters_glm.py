import sys
import h2o
import random
sys.path.insert(1, '../../../')
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

def test_glm_effective_parameters():
    if False:
        while True:
            i = 10
    cars = h2o.import_file(path=pyunit_utils.locate('smalldata/junit/cars_20mpg.csv'))
    predictors = ['displacement', 'power', 'weight', 'acceleration', 'year']
    response_col = 'economy_20mpg'
    family = 'binomial'
    cars[response_col] = cars[response_col].asfactor()
    nfolds = random.randint(3, 10)
    glm = H2OGeneralizedLinearEstimator(nfolds=nfolds, family=family)
    glm.train(x=predictors, y=response_col, training_frame=cars)
    assert glm.parms['fold_assignment']['input_value'] == 'AUTO'
    assert glm.parms['fold_assignment']['actual_value'] == 'Random'
    try:
        h2o.rapids('(setproperty "{}" "{}")'.format('sys.ai.h2o.algos.evaluate_auto_model_parameters', 'false'))
        glm = H2OGeneralizedLinearEstimator(nfolds=nfolds, family=family)
        glm.train(x=predictors, y=response_col, training_frame=cars)
        assert glm.parms['fold_assignment']['input_value'] == 'AUTO'
        assert glm.parms['fold_assignment']['actual_value'] == 'AUTO'
    finally:
        h2o.rapids('(setproperty "{}" "{}")'.format('sys.ai.h2o.algos.evaluate_auto_model_parameters', 'true'))
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_glm_effective_parameters)
else:
    test_glm_effective_parameters()