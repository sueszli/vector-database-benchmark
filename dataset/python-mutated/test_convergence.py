import math
import numpy as np
import pytest
import ray
from ray import train, tune
from ray.tune.stopper import ExperimentPlateauStopper
from ray.tune.search import ConcurrencyLimiter
import unittest

def loss(config):
    if False:
        print('Hello World!')
    x = config.get('x')
    train.report({'loss': x ** 2})

class ConvergenceTest(unittest.TestCase):
    """Test convergence in gaussian process."""

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            i = 10
            return i + 15
        ray.init(local_mode=False, num_cpus=1, num_gpus=0)

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            print('Hello World!')
        ray.shutdown()

    def _testConvergence(self, searcher, top=3, patience=20):
        if False:
            return 10
        space = {'x': tune.uniform(0, 20)}
        resources_per_trial = {'cpu': 1, 'gpu': 0}
        analysis = tune.run(loss, metric='loss', mode='min', stop=ExperimentPlateauStopper(metric='loss', top=top, patience=patience), search_alg=searcher, config=space, num_samples=max(100, patience), resources_per_trial=resources_per_trial, raise_on_failed_trial=False, fail_fast=True, reuse_actors=True, verbose=1)
        print(f"Num trials: {len(analysis.trials)}. Best result: {analysis.best_config['x']}")
        return analysis

    @unittest.skip('ax warm start tests currently failing (need to upgrade ax)')
    def testConvergenceAx(self):
        if False:
            print('Hello World!')
        from ray.tune.search.ax import AxSearch
        np.random.seed(0)
        searcher = AxSearch()
        analysis = self._testConvergence(searcher, patience=10)
        assert math.isclose(analysis.best_config['x'], 0, abs_tol=1e-05)

    def testConvergenceBayesOpt(self):
        if False:
            for i in range(10):
                print('nop')
        from ray.tune.search.bayesopt import BayesOptSearch
        np.random.seed(0)
        searcher = BayesOptSearch(random_search_steps=10)
        searcher.repeat_float_precision = 5
        searcher = ConcurrencyLimiter(searcher, 1)
        analysis = self._testConvergence(searcher, patience=100)
        assert len(analysis.trials) < 50
        assert math.isclose(analysis.best_config['x'], 0, abs_tol=1e-05)

    def testConvergenceBlendSearch(self):
        if False:
            return 10
        from ray.tune.search.flaml import BlendSearch
        np.random.seed(0)
        searcher = BlendSearch()
        analysis = self._testConvergence(searcher, patience=200)
        assert math.isclose(analysis.best_config['x'], 0, abs_tol=0.01)

    def testConvergenceCFO(self):
        if False:
            while True:
                i = 10
        from ray.tune.search.flaml import CFO
        np.random.seed(0)
        searcher = CFO()
        analysis = self._testConvergence(searcher, patience=200)
        assert math.isclose(analysis.best_config['x'], 0, abs_tol=0.01)

    def testConvergenceDragonfly(self):
        if False:
            return 10
        from ray.tune.search.dragonfly import DragonflySearch
        np.random.seed(0)
        searcher = DragonflySearch(domain='euclidean', optimizer='bandit')
        analysis = self._testConvergence(searcher)
        assert len(analysis.trials) < 100
        assert math.isclose(analysis.best_config['x'], 0, abs_tol=1e-05)

    def testConvergenceHEBO(self):
        if False:
            i = 10
            return i + 15
        from ray.tune.search.hebo import HEBOSearch
        np.random.seed(0)
        searcher = HEBOSearch()
        analysis = self._testConvergence(searcher)
        assert len(analysis.trials) < 100
        assert math.isclose(analysis.best_config['x'], 0, abs_tol=0.01)

    def testConvergenceHyperopt(self):
        if False:
            i = 10
            return i + 15
        from ray.tune.search.hyperopt import HyperOptSearch
        np.random.seed(0)
        searcher = HyperOptSearch(random_state_seed=1234)
        analysis = self._testConvergence(searcher, patience=50, top=5)
        assert math.isclose(analysis.best_config['x'], 0, abs_tol=0.01)

    def testConvergenceNevergrad(self):
        if False:
            while True:
                i = 10
        from ray.tune.search.nevergrad import NevergradSearch
        import nevergrad as ng
        np.random.seed(0)
        searcher = NevergradSearch(optimizer=ng.optimizers.PSO)
        analysis = self._testConvergence(searcher, patience=50, top=5)
        assert math.isclose(analysis.best_config['x'], 0, abs_tol=0.001)

    def testConvergenceOptuna(self):
        if False:
            return 10
        from ray.tune.search.optuna import OptunaSearch
        np.random.seed(1)
        searcher = OptunaSearch(seed=1)
        analysis = self._testConvergence(searcher, top=5)
        assert len(analysis.trials) < 100
        assert math.isclose(analysis.best_config['x'], 0, abs_tol=0.1)

    def testConvergenceSkOpt(self):
        if False:
            while True:
                i = 10
        from ray.tune.search.skopt import SkOptSearch
        np.random.seed(0)
        searcher = SkOptSearch()
        analysis = self._testConvergence(searcher)
        assert len(analysis.trials) < 100
        assert math.isclose(analysis.best_config['x'], 0, abs_tol=0.001)

    def testConvergenceZoopt(self):
        if False:
            print('Hello World!')
        from ray.tune.search.zoopt import ZOOptSearch
        np.random.seed(0)
        searcher = ZOOptSearch(budget=100)
        analysis = self._testConvergence(searcher)
        assert len(analysis.trials) < 100
        assert math.isclose(analysis.best_config['x'], 0, abs_tol=0.001)
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))