import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator as glm

def grab_lambda_values_alpha_best():
    if False:
        i = 10
        return i + 15
    boston = h2o.import_file(pyunit_utils.locate('smalldata/gbm_test/BostonHousing.csv'))
    predictors = boston.columns[:-1]
    response = 'medv'
    boston['chas'] = boston['chas'].asfactor()
    (train, valid) = boston.split_frame(ratios=[0.8], seed=1234)
    boston_glm = glm(lambda_search=True, seed=1234, cold_start=True)
    boston_glm.train(x=predictors, y=response, training_frame=train, validation_frame=valid)
    r = glm.getGLMRegularizationPath(boston_glm)
    assert glm.getLambdaBest(boston_glm) >= r['lambdas'][len(r['lambdas']) - 1] and glm.getLambdaBest(boston_glm) <= r['lambdas'][0], 'Error in lambda best extraction'
    assert glm.getLambdaMin(boston_glm) <= r['lambdas'][len(r['lambdas']) - 1], 'Error in lambda min extraction'
    assert glm.getLambdaMax(boston_glm) == r['lambdas'][0], 'Error in lambda max extraction'
    assert glm.getAlphaBest(boston_glm) == boston_glm._model_json['output']['alpha_best'], 'Error in alpha best extraction'
    boston_glm2 = glm(lambda_search=False, seed=1234)
    boston_glm2.train(x=predictors, y=response, training_frame=train, validation_frame=valid)
    try:
        glm.getLambdaMax(boston_glm2)
        assert False, 'glm.getLambdaMax(model) should have thrown an error but did not!'
    except Exception as ex:
        print(ex)
        temp = str(ex)
        assert 'getLambdaMax(model) can only be called when lambda_search=True' in temp
        print('grab_lambda_values) test completed!')
if __name__ == '__main__':
    pyunit_utils.standalone_test(grab_lambda_values_alpha_best)
else:
    grab_lambda_values_alpha_best()