import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator as glm

def glm_alpha_lambda_arrays():
    if False:
        while True:
            i = 10
    train = h2o.import_file(path=pyunit_utils.locate('smalldata/glm_test/binomial_20_cols_10KRows.csv'))
    for ind in range(10):
        train[ind] = train[ind].asfactor()
    train['C21'] = train['C21'].asfactor()
    frames = train.split_frame(ratios=[0.8], seed=12345)
    d = frames[0]
    d_test = frames[1]
    regKeys = ['alphas', 'lambdas', 'explained_deviance_valid', 'explained_deviance_train']
    mLVal = glm(family='binomial', alpha=[0.1, 0.5], lambda_search=True, solver='COORDINATE_DESCENT', nlambdas=3)
    mLVal.train(training_frame=d, x=list(range(20)), y=20, validation_frame=d_test)
    rVal = glm.getGLMRegularizationPath(mLVal)
    best_submodel_indexVal = mLVal._model_json['output']['best_submodel_index']
    m2Val = glm.makeGLMModel(model=mLVal, coefs=rVal['coefficients'][best_submodel_indexVal])
    dev1Val = rVal['explained_deviance_valid'][best_submodel_indexVal]
    p2Val = m2Val.model_performance(d_test)
    dev2Val = 1 - p2Val.residual_deviance() / p2Val.null_deviance()
    assert abs(dev1Val - dev2Val) < 1e-06
    for l in range(0, len(rVal['lambdas'])):
        m = glm(family='binomial', alpha=[rVal['alphas'][l]], Lambda=rVal['lambdas'][l], solver='COORDINATE_DESCENT')
        m.train(training_frame=d, x=list(range(20)), y=20, validation_frame=d_test)
        mr = glm.getGLMRegularizationPath(m)
        p = m.model_performance(d_test)
        cs = rVal['coefficients'][l]
        cs_norm = rVal['coefficients_std'][l]
        print('Comparing submodel index {0}'.format(l))
        pyunit_utils.assertEqualCoeffDicts(cs, m.coef(), tol=0.1)
        pyunit_utils.assertEqualCoeffDicts(cs_norm, m.coef_norm(), tol=0.1)
        pyunit_utils.assertEqualRegPaths(regKeys, rVal, l, mr, tol=0.001)
        dVal = 1 - p.residual_deviance() / p.null_deviance()
        if l == best_submodel_indexVal:
            pyunit_utils.assertEqualModelMetrics(m._model_json['output']['validation_metrics'], mLVal._model_json['output']['validation_metrics'], tol=0.01)
        else:
            assert dVal <= dev2Val, 'Best submodel does not have highest explained deviance_valid for submodel: !'.format(l)
if __name__ == '__main__':
    pyunit_utils.standalone_test(glm_alpha_lambda_arrays)
else:
    glm_alpha_lambda_arrays()