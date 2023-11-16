import sys
import time
sys.path.insert(1, '../../../')
import h2o
from h2o.estimators.estimator_base import H2OEstimator
from tests import pyunit_utils as pu

class DummyEstimator(H2OEstimator):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(DummyEstimator, self).__init__()
        self._parms = {}

    def _train(self, parms, verbose=False):
        if False:
            print('Hello World!')
        pass

def test_basic_estimator_preparation_perf_with_x():
    if False:
        while True:
            i = 10
    dummy = DummyEstimator()
    shape = (5, 100000)
    data_start = time.time()
    names = ['Col_' + str(n) for n in range(shape[1])]
    y = names[len(names) // 2]
    x = [n for (i, n) in enumerate(names) if i % 2]
    train_fr = h2o.H2OFrame({n: list(range(shape[0])) for n in names})
    data_duration = time.time() - data_start
    print('data preparation/upload took {}s'.format(data_duration))
    training_start = time.time()
    dummy.train(x=x, y=y, training_frame=train_fr, validation_frame=train_fr)
    training_duration = time.time() - training_start
    print('training preparation took {}s'.format(training_duration))
    assert training_duration < 10

def test_basic_estimator_preparation_perf_with_ignored_columns():
    if False:
        i = 10
        return i + 15
    dummy = DummyEstimator()
    shape = (5, 100000)
    data_start = time.time()
    names = ['Col_' + str(n) for n in range(shape[1])]
    y = names[len(names) // 2]
    ignored = [n for (i, n) in enumerate(names) if i % 2]
    train_fr = h2o.H2OFrame({n: list(range(shape[0])) for n in names})
    data_duration = time.time() - data_start
    print('data preparation/upload took {}s'.format(data_duration))
    training_start = time.time()
    dummy.train(y=y, training_frame=train_fr, validation_frame=train_fr, ignored_columns=ignored)
    training_duration = time.time() - training_start
    print('training preparation took {}s'.format(training_duration))
    assert training_duration < 10
pu.run_tests([test_basic_estimator_preparation_perf_with_x, test_basic_estimator_preparation_perf_with_ignored_columns])