"""Tests for MirroredStrategy."""
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import test

def get_gpus():
    if False:
        while True:
            i = 10
    gpus = context.context().list_logical_devices('GPU')
    actual_gpus = []
    for gpu in gpus:
        if 'job' in gpu.name:
            actual_gpus.append(gpu.name)
    return actual_gpus

@combinations.generate(combinations.combine(distribution=[combinations.NamedDistribution('Mirrored', lambda : mirrored_strategy.MirroredStrategy(get_gpus()), required_gpus=1)], mode=['eager']))
class RemoteSingleWorkerMirroredStrategyEager(multi_worker_test_base.SingleWorkerTestBaseEager, strategy_test_lib.RemoteSingleWorkerMirroredStrategyBase):

    def _get_num_gpus(self):
        if False:
            return 10
        return len(get_gpus())

    def testNumReplicasInSync(self, distribution):
        if False:
            print('Hello World!')
        self._testNumReplicasInSync(distribution)

    def testMinimizeLoss(self, distribution):
        if False:
            print('Hello World!')
        self._testMinimizeLoss(distribution)

    def testDeviceScope(self, distribution):
        if False:
            return 10
        self._testDeviceScope(distribution)

    def testMakeInputFnIteratorWithDataset(self, distribution):
        if False:
            i = 10
            return i + 15
        self._testMakeInputFnIteratorWithDataset(distribution)

    def testMakeInputFnIteratorWithCallable(self, distribution):
        if False:
            while True:
                i = 10
        self._testMakeInputFnIteratorWithCallable(distribution)
if __name__ == '__main__':
    test.main()