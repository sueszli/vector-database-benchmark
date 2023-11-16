import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.gbm import H2OGradientBoostingEstimator
import math

def cv_nfolds_sd_check():
    if False:
        return 10
    prostate = h2o.import_file(path=pyunit_utils.locate('smalldata/logreg/prostate.csv'))
    prostate[1] = prostate[1].asfactor()
    prostate.summary()
    prostate_gbm = H2OGradientBoostingEstimator(nfolds=4, distribution='bernoulli')
    prostate_gbm.train(x=list(range(2, 9)), y=1, training_frame=prostate)
    prostate_gbm.show()
    prostate_gbm.model_performance(xval=True)
    meanCol = pyunit_utils.extract_col_value_H2OTwoDimTable(prostate_gbm._model_json['output']['cross_validation_metrics_summary'], 'mean')
    stdCol = pyunit_utils.extract_col_value_H2OTwoDimTable(prostate_gbm._model_json['output']['cross_validation_metrics_summary'], 'sd')
    cv1 = pyunit_utils.extract_col_value_H2OTwoDimTable(prostate_gbm._model_json['output']['cross_validation_metrics_summary'], 'cv_1_valid')
    cv2 = pyunit_utils.extract_col_value_H2OTwoDimTable(prostate_gbm._model_json['output']['cross_validation_metrics_summary'], 'cv_2_valid')
    cv3 = pyunit_utils.extract_col_value_H2OTwoDimTable(prostate_gbm._model_json['output']['cross_validation_metrics_summary'], 'cv_3_valid')
    cv4 = pyunit_utils.extract_col_value_H2OTwoDimTable(prostate_gbm._model_json['output']['cross_validation_metrics_summary'], 'cv_4_valid')
    cvVals = [cv1, cv2, cv3, cv4]
    assertMeanSDCalculation(meanCol, stdCol, cvVals)

def assertMeanSDCalculation(meanCol, stdCol, cvVals, tol=1e-06):
    if False:
        for i in range(10):
            print('nop')
    '\n    For performance metrics calculated by cross-validation, we take the actual values and calculated the mean and\n    variance manually.  Next we compare the two and make sure they are equal\n    \n    :param meanCol: mean values over all nfolds\n    :param stdCol: std values over all nfolds\n    :param cvVals: actual values over all nfolds\n    :param tol: error tolerance\n    :return: error if the two sets of values are different.\n    '
    nfolds = len(cvVals)
    nItems = len(meanCol)
    oneOverNm1 = 1.0 / (nfolds - 1.0)
    for itemIndex in range(nItems):
        xsum = 0
        xsumSquare = 0
        for foldIndex in range(nfolds):
            temp = float(cvVals[foldIndex][itemIndex])
            xsum += temp
            xsumSquare += temp * temp
        xmean = xsum / nfolds
        assert abs(xmean - float(meanCol[itemIndex])) < tol, 'Expected mean: {0}, Actual mean: {1}'.format(xmean, float(meanCol[itemIndex]))
        xstd = math.sqrt((xsumSquare - nfolds * xmean * xmean) * oneOverNm1)
        assert abs(xstd - float(stdCol[itemIndex])) < tol, 'Expected SD: {0}, Actual SD: {1}'.format(xstd, float(stdCol[itemIndex]))
if __name__ == '__main__':
    pyunit_utils.standalone_test(cv_nfolds_sd_check)
else:
    cv_nfolds_sd_check()