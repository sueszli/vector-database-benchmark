import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.random_forest import H2ORandomForestEstimator
import numpy as np
from sklearn import ensemble
from sklearn.metrics import roc_auc_score

def smallcatRF():
    if False:
        for i in range(10):
            print('nop')
    alphabet = h2o.import_file(path=pyunit_utils.locate('smalldata/gbm_test/alphabet_cattest.csv'))
    alphabet['y'] = alphabet['y'].asfactor()
    trainData = np.loadtxt(pyunit_utils.locate('smalldata/gbm_test/alphabet_cattest.csv'), delimiter=',', skiprows=1, converters={0: lambda s: ord(s.decode().split('"')[1])})
    trainDataResponse = trainData[:, 1]
    trainDataFeatures = trainData[:, 0]
    rf_h2o = H2ORandomForestEstimator(ntrees=1, max_depth=1, nbins=100)
    rf_h2o.train(x='X', y='y', training_frame=alphabet)
    rf_sci = ensemble.RandomForestClassifier(n_estimators=1, criterion='entropy', max_depth=1)
    rf_sci.fit(trainDataFeatures[:, np.newaxis], trainDataResponse)
    rf_perf = rf_h2o.model_performance(alphabet)
    auc_h2o = rf_perf.auc()
    auc_sci = roc_auc_score(trainDataResponse, rf_sci.predict_proba(trainDataFeatures[:, np.newaxis])[:, 1])
    assert auc_h2o >= auc_sci, 'h2o (auc) performance degradation, with respect to scikit'
if __name__ == '__main__':
    pyunit_utils.standalone_test(smallcatRF)
else:
    smallcatRF()