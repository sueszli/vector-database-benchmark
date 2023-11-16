import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator as glm
import numpy as np

def test_define_dataset():
    if False:
        while True:
            i = 10
    family = 'gaussian'
    nrow = 10000
    ncol = 10
    realFrac = 0.5
    intFrac = 0
    enumFrac = 0.5
    missing_fraction = 0
    factorRange = 5
    numericRange = 10
    targetFactor = 4
    glmDataSet = generate_dataset(family, nrow, ncol, realFrac, intFrac, enumFrac, missing_fraction, factorRange, numericRange, targetFactor)
    assert glmDataSet.nrow == nrow, 'Dataset number of row: {0}, expected number of row: {1}'.format(glmDataSet.nrow, nrow)
    assert glmDataSet.ncol == 1 + ncol, 'Dataset number of row: {0}, expected number of row: {1}'.format(glmDataSet.ncol, 1 + ncol)

def generate_dataset(family, nrow, ncol, realFrac, intFrac, enumFrac, missingFrac, factorRange, numericRange, targetFactor):
    if False:
        print('Hello World!')
    if family == 'binomial':
        responseFactor = 2
    elif family == 'multinomial' or family == 'ordinal':
        responseFactor = targetFactor
    else:
        responseFactor = 1
    trainData = random_dataset(nrow, ncol, realFrac=realFrac, intFrac=intFrac, enumFrac=enumFrac, factorR=factorRange, integerR=numericRange, responseFactor=responseFactor, misFrac=missingFrac)
    if family == 'poisson':
        trainData['response'] = trainData['response'] + numericRange
    myX = trainData.names
    myY = 'response'
    myX.remove(myY)
    m = glm(family=family, max_iterations=1, interactions=['C1', 'C2'], tweedie_link_power=2, tweedie_variance_power=0.4)
    m.train(training_frame=trainData, x=myX, y=myY)
    r = glm.getGLMRegularizationPath(m)
    coeffDict = r['coefficients'][0]
    coeffLen = len(coeffDict)
    randCoeffVals = np.random.uniform(low=-3, high=3, size=coeffLen).tolist()
    keyset = coeffDict.keys()
    count = 0
    for key in keyset:
        coeffDict[key] = randCoeffVals[count]
        count = count + 1
    m2 = glm.makeGLMModel(model=m, coefs=coeffDict)
    f2 = m2.predict(trainData)
    finalDataset = trainData[myX]
    finalDataset = finalDataset.cbind(f2[0])
    finalDataset.set_name(col=finalDataset.ncols - 1, name='response')
    return finalDataset

def random_dataset(nrow, ncol, realFrac=0.4, intFrac=0.3, enumFrac=0.3, factorR=10, integerR=100, responseFactor=1, misFrac=0.01, randSeed=None):
    if False:
        while True:
            i = 10
    fractions = dict()
    fractions['real_fraction'] = realFrac
    fractions['categorical_fraction'] = enumFrac
    fractions['integer_fraction'] = intFrac
    fractions['time_fraction'] = 0
    fractions['string_fraction'] = 0
    fractions['binary_fraction'] = 0
    df = h2o.create_frame(rows=nrow, cols=ncol, missing_fraction=misFrac, has_response=True, response_factors=responseFactor, factors=factorR, integer_range=integerR, real_range=integerR, seed=randSeed, **fractions)
    print(df.types)
    return df
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_define_dataset)
else:
    test_define_dataset()