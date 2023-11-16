from builtins import range
import sys, os
sys.path.insert(1, os.path.join('..', '..', '..'))
import h2o
from tests import pyunit_utils
import numpy as np
from sklearn import ensemble
from sklearn.metrics import roc_auc_score
from h2o.estimators.gbm import H2OGradientBoostingEstimator

def bernoulli_gbm():
    if False:
        print('Hello World!')
    prostate_train = h2o.import_file(path=pyunit_utils.locate('smalldata/logreg/prostate_train.csv'))
    prostate_train['CAPSULE'] = prostate_train['CAPSULE'].asfactor()
    trainData = np.loadtxt(pyunit_utils.locate('smalldata/logreg/prostate_train.csv'), delimiter=',', skiprows=1)
    trainDataResponse = trainData[:, 0]
    trainDataFeatures = trainData[:, 1:]
    ntrees = 100
    learning_rate = 0.1
    depth = 5
    min_rows = 10
    gbm_h2o = H2OGradientBoostingEstimator(ntrees=ntrees, learn_rate=learning_rate, max_depth=depth, min_rows=min_rows, distribution='bernoulli')
    gbm_h2o.train(x=list(range(1, prostate_train.ncol)), y='CAPSULE', training_frame=prostate_train)
    gbm_sci = ensemble.GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=ntrees, max_depth=depth, min_samples_leaf=min_rows, max_features=None)
    gbm_sci.fit(trainDataFeatures, trainDataResponse)
    prostate_test = h2o.import_file(path=pyunit_utils.locate('smalldata/logreg/prostate_test.csv'))
    prostate_test['CAPSULE'] = prostate_test['CAPSULE'].asfactor()
    testData = np.loadtxt(pyunit_utils.locate('smalldata/logreg/prostate_test.csv'), delimiter=',', skiprows=1)
    testDataResponse = testData[:, 0]
    testDataFeatures = testData[:, 1:]
    auc_sci = roc_auc_score(testDataResponse, gbm_sci.predict_proba(testDataFeatures)[:, 1])
    gbm_perf = gbm_h2o.model_performance(prostate_test)
    auc_h2o = gbm_perf.auc()
    print(auc_h2o, auc_sci)
    assert auc_h2o >= auc_sci, 'h2o (auc) performance degradation, with respect to scikit'
if __name__ == '__main__':
    pyunit_utils.standalone_test(bernoulli_gbm)
else:
    bernoulli_gbm()