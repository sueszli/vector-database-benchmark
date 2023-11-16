import numpy as np
import os
import shutil
import unittest
import ray
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import framework_iterator
from ray.tune.registry import get_trainable_cls
(tf1, tf, tfv) = try_import_tf()
(torch, _) = try_import_torch()
RLMODULE_SUPPORTED_ALGOS = {'PPO'}

def save_test(alg_name, framework='tf', multi_agent=False):
    if False:
        for i in range(10):
            print('nop')
    config = get_trainable_cls(alg_name).get_default_config().framework(framework).checkpointing(export_native_model_files=True)
    if alg_name in RLMODULE_SUPPORTED_ALGOS:
        config = config.experimental(_enable_new_api_stack=False)
    if 'DDPG' in alg_name or 'SAC' in alg_name:
        config.environment('Pendulum-v1')
        algo = config.build()
        test_obs = np.array([[0.1, 0.2, 0.3]])
    else:
        if multi_agent:
            config.multi_agent(policies={'pol1', 'pol2'}, policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: 'pol1' if agent_id == 'agent1' else 'pol2')
            config.environment(MultiAgentCartPole, env_config={'num_agents': 2})
        else:
            config.environment('CartPole-v1')
        algo = config.build()
        test_obs = np.array([[0.1, 0.2, 0.3, 0.4]])
    export_dir = os.path.join(ray._private.utils.get_user_temp_dir(), 'export_dir_%s' % alg_name)
    algo.train()
    print('Exporting algo checkpoint', alg_name, export_dir)
    export_dir = algo.save(export_dir).checkpoint.path
    model_dir = os.path.join(export_dir, 'policies', 'pol1' if multi_agent else DEFAULT_POLICY_ID, 'model')
    if framework == 'torch':
        filename = os.path.join(model_dir, 'model.pt')
        model = torch.load(filename)
        assert model
        results = model(input_dict={'obs': torch.from_numpy(test_obs)}, state=[torch.tensor(0)], seq_lens=torch.tensor(0))
        assert len(results) == 2
        assert results[0].shape == (1, 2)
        assert results[1] == [torch.tensor(0)]
    else:
        model = tf.saved_model.load(model_dir)
        assert model
        results = model(tf.convert_to_tensor(test_obs, dtype=tf.float32))
        assert len(results) == 2
        assert results[0].shape == (1, 2)
        assert results[1].shape == (1, 1)
    shutil.rmtree(export_dir)

class TestAlgorithmSave(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            while True:
                i = 10
        ray.init(num_cpus=4)

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        ray.shutdown()

    def test_save_appo_multi_agent(self):
        if False:
            i = 10
            return i + 15
        for fw in framework_iterator():
            save_test('APPO', fw, multi_agent=True)

    def test_save_ppo(self):
        if False:
            print('Hello World!')
        for fw in framework_iterator():
            save_test('PPO', fw)
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))