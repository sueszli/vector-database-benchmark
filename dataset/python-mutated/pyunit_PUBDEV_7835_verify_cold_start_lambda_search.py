import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

def grab_lambda_min():
    if False:
        for i in range(10):
            print('nop')
    boston = h2o.import_file(pyunit_utils.locate('smalldata/gbm_test/BostonHousing.csv'))
    predictors = boston.columns[:-1]
    response = 'medv'
    boston['chas'] = boston['chas'].asfactor()
    (train, valid) = boston.split_frame(ratios=[0.8], seed=1234)
    boston_glm = H2OGeneralizedLinearEstimator(lambda_search=True, seed=1234, cold_start=True)
    boston_glm.train(x=predictors, y=response, training_frame=train, validation_frame=valid)
    r = H2OGeneralizedLinearEstimator.getGLMRegularizationPath(boston_glm)
    for l in range(0, len(r['lambdas'])):
        m = H2OGeneralizedLinearEstimator(alpha=[r['alphas'][l]], Lambda=r['lambdas'][l], solver='COORDINATE_DESCENT')
        m.train(x=predictors, y=response, training_frame=train, validation_frame=valid)
        cs = r['coefficients'][l]
        cs_norm = r['coefficients_std'][l]
        print('comparing coefficients for submodel {0}'.format(l))
        pyunit_utils.assertEqualCoeffDicts(cs, m.coef(), tol=1e-06)
        pyunit_utils.assertEqualCoeffDicts(cs_norm, m.coef_norm(), tol=1e-06)
if __name__ == '__main__':
    pyunit_utils.standalone_test(grab_lambda_min)
else:
    grab_lambda_min()