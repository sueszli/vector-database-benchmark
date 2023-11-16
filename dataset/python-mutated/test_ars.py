import unittest
from rllib_ars.ars.ars import ARSConfig
import ray
from ray.rllib.utils.test_utils import check_compute_single_action, framework_iterator

class TestARS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        ray.init()

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        ray.shutdown()

    def test_ars_compilation(self):
        if False:
            return 10
        'Test whether an ARSAlgorithm can be built on all frameworks.'
        config = ARSConfig()
        config.training(model={'fcnet_hiddens': [10], 'fcnet_activation': None}, noise_size=2500000)
        config.evaluation(evaluation_interval=1, evaluation_num_workers=1)
        num_iterations = 2
        for _ in framework_iterator(config):
            algo = config.build(env='CartPole-v1')
            for i in range(num_iterations):
                results = algo.train()
                print(results)
            check_compute_single_action(algo)
            algo.stop()
if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main(['-v', __file__]))