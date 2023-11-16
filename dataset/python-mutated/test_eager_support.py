import unittest
import ray
from ray import air
from ray import tune
from ray.rllib.utils.framework import try_import_tf
from ray.tune.registry import get_trainable_cls
(tf1, tf, tfv) = try_import_tf()

def check_support(alg, config, test_eager=False, test_trace=True):
    if False:
        print('Hello World!')
    config['framework'] = 'tf2'
    config['log_level'] = 'ERROR'
    for cont in [True, False]:
        if cont and alg == 'DQN':
            continue
        if cont:
            config['env'] = 'Pendulum-v1'
        else:
            config['env'] = 'CartPole-v1'
        a = get_trainable_cls(alg)
        if test_eager:
            print('tf-eager: alg={} cont.act={}'.format(alg, cont))
            config['eager_tracing'] = False
            tune.Tuner(a, param_space=config, run_config=air.RunConfig(stop={'training_iteration': 1}, verbose=1)).fit()
        if test_trace:
            config['eager_tracing'] = True
            print('tf-eager-tracing: alg={} cont.act={}'.format(alg, cont))
            tune.Tuner(a, param_space=config, run_config=air.RunConfig(stop={'training_iteration': 1}, verbose=1)).fit()

class TestEagerSupportPolicyGradient(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        ray.init(num_cpus=4)

    def tearDown(self):
        if False:
            return 10
        ray.shutdown()

    def test_dqn(self):
        if False:
            return 10
        check_support('DQN', {'num_workers': 0, 'num_steps_sampled_before_learning_starts': 0})

    def test_ppo(self):
        if False:
            for i in range(10):
                print('nop')
        check_support('PPO', {'num_workers': 0})

    def test_appo(self):
        if False:
            print('Hello World!')
        check_support('APPO', {'num_workers': 1, 'num_gpus': 0})

    def test_impala(self):
        if False:
            while True:
                i = 10
        check_support('IMPALA', {'num_workers': 1, 'num_gpus': 0}, test_eager=True)

class TestEagerSupportOffPolicy(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        ray.init(num_cpus=4)

    def tearDown(self):
        if False:
            while True:
                i = 10
        ray.shutdown()

    def test_dqn(self):
        if False:
            for i in range(10):
                print('nop')
        check_support('DQN', {'num_workers': 0, 'num_steps_sampled_before_learning_starts': 0})

    def test_sac(self):
        if False:
            return 10
        check_support('SAC', {'num_workers': 0, 'num_steps_sampled_before_learning_starts': 0})
if __name__ == '__main__':
    import sys
    if tfv == 2:
        print('\tskip due to tf==2.x')
        sys.exit(0)
    import pytest
    class_ = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(pytest.main(['-v', __file__ + ('' if class_ is None else '::' + class_)]))