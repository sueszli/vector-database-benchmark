from builtins import range
import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.gbm import H2OGradientBoostingEstimator

def model_summary():
    if False:
        i = 10
        return i + 15
    df = h2o.import_file(path=pyunit_utils.locate('smalldata/logreg/prostate.csv'))
    df.describe()
    train = df.drop('ID')
    vol = train['VOL']
    vol[vol == 0] = None
    gle = train['GLEASON']
    gle[gle == 0] = None
    train['CAPSULE'] = train['CAPSULE'].asfactor()
    train.describe()
    my_gbm = H2OGradientBoostingEstimator(ntrees=50, learn_rate=0.1, distribution='bernoulli')
    my_gbm.train(x=list(range(1, train.ncol)), y='CAPSULE', training_frame=train, validation_frame=train)
    summary = my_gbm.summary()
    metrics = [None] * 10
    for i in range(0, 10):
        metrics[i] = summary[i]
    for i in range(0, 10):
        assert metrics[i] == summary[i], 'Expected equal metrics in model summary and extracted model summary but got: {0}'.format(metrics[i])
if __name__ == '__main__':
    pyunit_utils.standalone_test(model_summary)
else:
    model_summary()