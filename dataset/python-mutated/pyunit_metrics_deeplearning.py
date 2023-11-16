from builtins import range
import sys, os
sys.path.insert(1, os.path.join('..', '..', '..'))
import h2o
from tests import pyunit_utils
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

def deep_learning_metrics_test():
    if False:
        for i in range(10):
            print('nop')
    df = h2o.import_file(path=pyunit_utils.locate('smalldata/logreg/prostate.csv'))
    df.drop('ID')
    df['CAPSULE'] = df['CAPSULE'].asfactor()
    vol = df['VOL']
    vol[vol == 0] = float('nan')
    r = vol.runif()
    train = df[r < 0.8]
    test = df[r >= 0.8]
    train.describe()
    train.head()
    train.tail()
    test.describe()
    test.head()
    test.tail()
    print('Train a Deeplearning model: ')
    dl = H2ODeepLearningEstimator(epochs=100, hidden=[10, 10, 10], loss='CrossEntropy')
    dl.train(x=list(range(2, train.ncol)), y='CAPSULE', training_frame=train)
    print('Binomial Model Metrics: ')
    print()
    dl.show()
    p = dl.model_performance(test)
    p.show()
if __name__ == '__main__':
    pyunit_utils.standalone_test(deep_learning_metrics_test)
else:
    deep_learning_metrics_test()