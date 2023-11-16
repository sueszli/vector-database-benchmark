from builtins import range
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
import math

def test_standardized_coeffs():
    if False:
        return 10
    print('Checking standardized coefficients for multinomials....')
    buildModelCheckStdCoeffs('smalldata/glm_test/multinomial_10_classes_10_cols_10000_Rows_train.csv', 'multinomial')
    print('Checking standardized coefficients for binomials....')
    buildModelCheckStdCoeffs('smalldata/glm_test/binomial_20_cols_10KRows.csv', 'binomial')
    print('Checking standardized coefficients for regression....')
    buildModelCheckStdCoeffs('smalldata/glm_test/gaussian_20cols_10000Rows.csv', 'gaussian')

def buildModelCheckStdCoeffs(training_fileName, family):
    if False:
        while True:
            i = 10
    training_data = h2o.import_file(pyunit_utils.locate(training_fileName))
    ncols = training_data.ncols
    Y = ncols - 1
    x = list(range(0, Y))
    enumCols = Y / 2
    if family == 'binomial' or family == 'multinomial':
        training_data[Y] = training_data[Y].asfactor()
    for ind in range(int(enumCols)):
        training_data[ind] = training_data[ind].asfactor()
    model1 = H2OGeneralizedLinearEstimator(family=family, standardize=True)
    model1.train(training_frame=training_data, x=x, y=Y)
    stdCoeff1 = model1.coef_norm()
    modelNS = H2OGeneralizedLinearEstimator(family=family, standardize=False)
    modelNS.train(training_frame=training_data, x=x, y=Y)
    coeffNSStandardized = modelNS.coef_norm()
    coeffNS = modelNS.coef()
    if family == 'multinomial':
        nclass = len(coeffNS)
        for cind in range(nclass):
            coeff1PerClass = coeffNSStandardized['std_coefs_class_' + str(cind)]
            coeff2PerClass = coeffNS['coefs_class_' + str(cind)]
            print('Comparing multinomial coefficients for class {0}'.format(cind))
            assert_coeffs_equal(coeff1PerClass, coeff2PerClass, training_data)
    else:
        assert_coeffs_equal(coeffNSStandardized, coeffNS, training_data)
    for ind in range(int(enumCols), Y):
        aver = training_data[ind].mean()
        sigma = 1.0 / math.sqrt(training_data[ind].var())
        training_data[ind] = (training_data[ind] - aver) * sigma
    model2 = H2OGeneralizedLinearEstimator(family=family, standardize=False)
    model2.train(training_frame=training_data, x=x, y=Y)
    coeff2 = model2.coef_norm()
    compare_coeffs_2_model(family, stdCoeff1, coeff2)
    coeff2Coef = model2.coef()
    compare_coeffs_2_model(family, coeff2, coeff2Coef, sameModel=True)

def compare_coeffs_2_model(family, coeff1, coeff2, sameModel=False):
    if False:
        return 10
    if family == 'multinomial':
        assert len(coeff1) == len(coeff2), 'Coefficient dictionary lengths are different.  One has length {0} while the other one has length {1}.'.format(len(coeff1), len(coeff2))
        if sameModel:
            coeff2CoefKeyChanged = dict()
            for index in range(len(coeff2)):
                key = 'coefs_class_' + str(index)
                coeff2CoefKeyChanged['std_' + key] = coeff2[key]
            coeff2 = coeff2CoefKeyChanged
        for name in coeff1.keys():
            pyunit_utils.equal_two_dicts(coeff1[name], coeff2[name])
    else:
        pyunit_utils.equal_two_dicts(coeff1, coeff2)

def assert_coeffs_equal(coeffStandard, coeff, training_data):
    if False:
        for i in range(10):
            print('nop')
    interceptOffset = 0
    for key in coeffStandard.keys():
        temp1 = coeffStandard[key]
        temp2 = coeff[key]
        if abs(temp1 - temp2) > 1e-06:
            if not key == 'Intercept':
                colIndex = int(float(key.split('C')[1])) - 1
                interceptOffset = interceptOffset + temp2 * training_data[colIndex].mean()[0, 0]
                temp2 = temp2 * math.sqrt(training_data[colIndex].var())
                assert abs(temp1 - temp2) < 1e-06, 'Expected: {0}, Actual: {1} at col: {2}'.format(temp2, temp1, key)
    temp1 = coeffStandard['Intercept']
    temp2 = coeff['Intercept'] + interceptOffset
    assert abs(temp1 - temp2) < 1e-06, 'Expected: {0}, Actual: {1} at Intercept'.format(temp2, temp1)
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_standardized_coeffs)
else:
    test_standardized_coeffs()