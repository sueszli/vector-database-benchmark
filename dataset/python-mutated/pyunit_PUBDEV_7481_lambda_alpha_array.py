import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator as glm

def glm_alpha_lambda_arrays():
    if False:
        i = 10
        return i + 15
    d = h2o.import_file(path=pyunit_utils.locate('smalldata/logreg/prostate.csv'))
    mL = glm(family='binomial', Lambda=[0.9, 0.5, 0.1], alpha=[0.1, 0.5, 0.9], solver='COORDINATE_DESCENT')
    mL.train(training_frame=d, x=[2, 3, 4, 5, 6, 7, 8], y=1)
    r = glm.getGLMRegularizationPath(mL)
    regKeys = ['alphas', 'lambdas', 'explained_deviance_valid', 'explained_deviance_train']
    best_submodel_index = mL._model_json['output']['best_submodel_index']
    m2 = glm.makeGLMModel(model=mL, coefs=r['coefficients'][best_submodel_index])
    dev1 = r['explained_deviance_train'][best_submodel_index]
    p2 = m2.model_performance(d)
    dev2 = 1 - p2.residual_deviance() / p2.null_deviance()
    assert abs(dev1 - dev2) < 1e-06
    for l in range(0, len(r['lambdas'])):
        m = glm(family='binomial', alpha=[r['alphas'][l]], Lambda=[r['lambdas'][l]], solver='COORDINATE_DESCENT')
        m.train(training_frame=d, x=[2, 3, 4, 5, 6, 7, 8], y=1)
        mr = glm.getGLMRegularizationPath(m)
        cs = r['coefficients'][l]
        cs_norm = r['coefficients_std'][l]
        diff = 0
        diff2 = 0
        for n in cs.keys():
            diff = max(diff, abs(cs[n] - m.coef()[n]))
            diff2 = max(diff2, abs(cs_norm[n] - m.coef_norm()[n]))
        assert diff < 0.01
        assert diff2 < 0.01
        p = m.model_performance(d)
        devm = 1 - p.residual_deviance() / p.null_deviance()
        devn = r['explained_deviance_train'][l]
        assert abs(devm - devn) < 0.0001
        pyunit_utils.assertEqualRegPaths(regKeys, r, l, mr, tol=1e-05)
        if l == best_submodel_index:
            pyunit_utils.assertEqualModelMetrics(m._model_json['output']['training_metrics'], mL._model_json['output']['training_metrics'], tol=1e-05)
        else:
            assert p.residual_deviance() >= p2.residual_deviance(), 'Best submodel does not have lowerest residual_deviance()!'
if __name__ == '__main__':
    pyunit_utils.standalone_test(glm_alpha_lambda_arrays)
else:
    glm_alpha_lambda_arrays()