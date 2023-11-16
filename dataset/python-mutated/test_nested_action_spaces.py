from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
import numpy as np
import os
import shutil
import tree
import unittest
import ray
from ray.rllib.algorithms.bc import BC
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.policy.sample_batch import convert_ma_batch_to_sample_batch
from ray.rllib.utils.test_utils import framework_iterator
SPACES = {'dict': Dict({'a': Dict({'aa': Box(-1.0, 1.0, shape=(3,)), 'ab': MultiDiscrete([4, 3])}), 'b': Discrete(3), 'c': Tuple([Box(0, 10, (2,), dtype=np.int32), Discrete(2)]), 'd': Box(0, 3, (), dtype=np.int64)}), 'tuple': Tuple([Tuple([Box(-1.0, 1.0, shape=(2,)), Discrete(3)]), MultiDiscrete([4, 3]), Dict({'a': Box(0, 100, (), dtype=np.int32), 'b': Discrete(2)})]), 'multidiscrete': MultiDiscrete([2, 3, 4]), 'intbox': Box(0, 100, (2,), dtype=np.int32)}

class NestedActionSpacesTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        ray.init(num_cpus=5)

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        ray.shutdown()

    def test_nested_action_spaces(self):
        if False:
            for i in range(10):
                print('nop')
        tmp_dir = os.popen('mktemp -d').read()[:-1]
        if not os.path.exists(tmp_dir):
            tmp_dir = ray._private.utils.tempfile.gettempdir() + tmp_dir[4:]
            assert os.path.exists(tmp_dir), f"'{tmp_dir}' not found!"
        config = PPOConfig().environment(RandomEnv).rollouts(num_rollout_workers=0).offline_data(output=tmp_dir, actions_in_input_normalized=True).evaluation(off_policy_estimation_methods={}).training(lr_schedule=None, train_batch_size=20, sgd_minibatch_size=20, num_sgd_iter=1, model={'fcnet_hiddens': [10]})
        for _ in framework_iterator(config):
            for (name, action_space) in SPACES.items():
                config.environment(env_config={'action_space': action_space})
                for flatten in [True, False]:
                    config.experimental(_disable_action_flattening=not flatten)
                    print(f'A={action_space} flatten={flatten}')
                    shutil.rmtree(config['output'])
                    algo = config.build()
                    algo.train()
                    algo.stop()
                    reader = JsonReader(inputs=config['output'], ioctx=algo.workers.local_worker().io_context)
                    sample_batch = reader.next()
                    sample_batch = convert_ma_batch_to_sample_batch(sample_batch)
                    if flatten:
                        assert isinstance(sample_batch['actions'], np.ndarray)
                        assert len(sample_batch['actions'].shape) == 2
                        assert sample_batch['actions'].shape[0] == len(sample_batch)
                    else:
                        tree.assert_same_structure(algo.get_policy().action_space_struct, sample_batch['actions'])
                    config['input'] = lambda ioctx: JsonReader(ioctx.config['input_config']['paths'], ioctx)
                    config['input_config'] = {'paths': config['output']}
                    config.output = None
                    bc = BC(config=config)
                    bc.train()
                    bc.stop()
                    config['output'] = tmp_dir
                    config['input'] = 'sampler'
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))