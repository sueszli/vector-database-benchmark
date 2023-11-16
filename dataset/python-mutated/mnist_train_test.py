"""Test for tfr mnist training example."""
from absl.testing import parameterized
from tensorflow.compiler.mlir.tfr.examples.mnist import mnist_train
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import test_util as distribute_test_util
from tensorflow.python.framework import test_util
strategies = [strategy_combinations.one_device_strategy, strategy_combinations.one_device_strategy_gpu, strategy_combinations.tpu_strategy]

class MnistTrainTest(test_util.TensorFlowTestCase, parameterized.TestCase):

    @combinations.generate(combinations.combine(strategy=strategies))
    def testMnistTrain(self, strategy):
        if False:
            for i in range(10):
                print('nop')
        accuracy = mnist_train.main(strategy)
        self.assertGreater(accuracy, 0.7, 'accuracy sanity check')
if __name__ == '__main__':
    distribute_test_util.main()