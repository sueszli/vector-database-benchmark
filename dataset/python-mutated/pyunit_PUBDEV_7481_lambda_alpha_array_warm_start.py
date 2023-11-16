import sys
sys.path.insert(1, '../../../')
import h2o
import math
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator as glm

def glm_alpha_lambda_arrays():
    if False:
        print('Hello World!')
    d = h2o.import_file(path=pyunit_utils.locate('smalldata/logreg/prostate.csv'))
    mL = glm(family='binomial', Lambda=[0.9, 0.5, 0.1], alpha=[0.1, 0.5, 0.9], solver='COORDINATE_DESCENT', cold_start=False)
    mL.train(training_frame=d, x=[2, 3, 4, 5, 6, 7, 8], y=1)
    r = glm.getGLMRegularizationPath(mL)
    regKeys = ['alphas', 'lambdas', 'explained_deviance_valid', 'explained_deviance_train']
    best_submodel_index = mL._model_json['output']['best_submodel_index']
    m2 = glm.makeGLMModel(model=mL, coefs=r['coefficients'][best_submodel_index])
    dev1 = r['explained_deviance_train'][best_submodel_index]
    p2 = m2.model_performance(d)
    dev2 = 1 - p2.residual_deviance() / p2.null_deviance()
    print(dev1, ' =?= ', dev2)
    assert abs(dev1 - dev2) < 1e-06
    responseMean = d[1].mean()
    initIntercept = math.log(responseMean / (1.0 - responseMean))
    startValInit = [0, 0, 0, 0, 0, 0, 0, initIntercept]
    startVal = [0, 0, 0, 0, 0, 0, 0, initIntercept]
    orderedCoeffNames = ['AGE', 'RACE', 'DPROS', 'DCAPS', 'PSA', 'VOL', 'GLEASON', 'Intercept']
    for l in range(0, len(r['lambdas'])):
        m = glm(family='binomial', alpha=[r['alphas'][l]], Lambda=[r['lambdas'][l]], solver='COORDINATE_DESCENT', startval=startVal)
        m.train(training_frame=d, x=[2, 3, 4, 5, 6, 7, 8], y=1)
        mr = glm.getGLMRegularizationPath(m)
        cs = r['coefficients'][l]
        cs_norm = r['coefficients_std'][l]
        pyunit_utils.assertEqualCoeffDicts(cs, m.coef(), tol=0.001)
        pyunit_utils.assertEqualCoeffDicts(cs_norm, m.coef_norm(), 0.001)
        if l + 1 < len(r['lambdas']) and r['alphas'][l] != r['alphas'][l + 1]:
            startVal = startValInit
        else:
            startVal = pyunit_utils.extractNextCoeff(cs_norm, orderedCoeffNames, startVal)
        p = m.model_performance(d)
        devm = 1 - p.residual_deviance() / p.null_deviance()
        devn = r['explained_deviance_train'][l]
        assert abs(devm - devn) < 0.0001
        pyunit_utils.assertEqualRegPaths(regKeys, r, l, mr, tol=0.0001)
        if l == best_submodel_index:
            pyunit_utils.assertEqualModelMetrics(m._model_json['output']['training_metrics'], mL._model_json['output']['training_metrics'], tol=0.0001)
        else:
            assert p.residual_deviance() >= p2.residual_deviance(), 'Best submodel does not have lowerest residual_deviance()!'
if __name__ == '__main__':
    pyunit_utils.standalone_test(glm_alpha_lambda_arrays)
else:
    glm_alpha_lambda_arrays()