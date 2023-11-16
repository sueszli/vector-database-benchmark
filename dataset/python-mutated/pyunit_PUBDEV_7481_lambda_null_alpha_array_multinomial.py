import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator as glm

def glm_alpha_array_lambda_null():
    if False:
        while True:
            i = 10
    d = h2o.import_file(path=pyunit_utils.locate('smalldata/covtype/covtype.20k.data'))
    mL = glm(family='multinomial', alpha=[0.1, 0.5, 0.9])
    d[54] = d[54].asfactor()
    mL.train(training_frame=d, x=list(range(0, 54)), y=54)
    r = glm.getGLMRegularizationPath(mL)
    regKeys = ['alphas', 'lambdas', 'explained_deviance_valid', 'explained_deviance_train']
    best_submodel_index = mL._model_json['output']['best_submodel_index']
    coefClassSet = ['coefs_class_0', 'coefs_class_1', 'coefs_class_2', 'coefs_class_3', 'coefs_class_4', 'coefs_class_5', 'coefs_class_6', 'coefs_class_7']
    coefClassSetNorm = ['std_coefs_class_0', 'std_coefs_class_1', 'std_coefs_class_2', 'std_coefs_class_3', 'std_coefs_class_4', 'std_coefs_class_5', 'std_coefs_class_6', 'std_coefs_class_7']
    for l in range(0, len(r['lambdas'])):
        m = glm(family='multinomial', alpha=[r['alphas'][l]], Lambda=[r['lambdas'][l]])
        m.train(training_frame=d, x=list(range(0, 54)), y=54)
        mr = glm.getGLMRegularizationPath(m)
        cs = r['coefficients'][l]
        cs_norm = r['coefficients_std'][l]
        pyunit_utils.assertCoefEqual(cs, m.coef(), coefClassSet, tol=1e-05)
        pyunit_utils.assertCoefEqual(cs_norm, m.coef_norm(), coefClassSetNorm, tol=1e-05)
        devm = 1 - m.residual_deviance() / m.null_deviance()
        devn = r['explained_deviance_train'][l]
        assert abs(devm - devn) < 0.0001
        pyunit_utils.assertEqualRegPaths(regKeys, r, l, mr)
        if l == best_submodel_index:
            pyunit_utils.assertEqualModelMetrics(m._model_json['output']['training_metrics'], mL._model_json['output']['training_metrics'], tol=0.01)
        else:
            assert m.logloss() >= mL.logloss(), 'Best submodel does not have lowerest logloss()!'
if __name__ == '__main__':
    pyunit_utils.standalone_test(glm_alpha_array_lambda_null)
else:
    glm_alpha_array_lambda_null()