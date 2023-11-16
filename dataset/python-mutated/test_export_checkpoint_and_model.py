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
RLMODULE_SUPPORTED_ALGOS = {'APPO', 'IMPALA', 'PPO'}

def export_test(alg_name, framework='tf', multi_agent=False, tf_expected_to_work=True):
    if False:
        i = 10
        return i + 15
    cls = get_trainable_cls(alg_name)
    config = cls.get_default_config()
    if alg_name in RLMODULE_SUPPORTED_ALGOS:
        config = config.experimental(_enable_new_api_stack=False)
    config.framework(framework)
    config.checkpointing(export_native_model_files=True)
    if 'SAC' in alg_name:
        algo = config.build(env='Pendulum-v1')
        test_obs = np.array([[0.1, 0.2, 0.3]])
    else:
        if multi_agent:
            config.multi_agent(policies={'pol1', 'pol2'}, policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: 'pol1' if agent_id == 'agent1' else 'pol2').environment(MultiAgentCartPole, env_config={'num_agents': 2})
        else:
            config.environment('CartPole-v1')
        algo = config.build()
        test_obs = np.array([[0.1, 0.2, 0.3, 0.4]])
    export_dir = os.path.join(ray._private.utils.get_user_temp_dir(), 'export_dir_%s' % alg_name)
    print('Exporting policy checkpoint', alg_name, export_dir)
    if multi_agent:
        algo.export_policy_checkpoint(export_dir, policy_id='pol1')
    else:
        algo.export_policy_checkpoint(export_dir, policy_id=DEFAULT_POLICY_ID)
    if framework == 'torch':
        model = torch.load(os.path.join(export_dir, 'model', 'model.pt'))
        assert model
        results = model(input_dict={'obs': torch.from_numpy(test_obs)}, state=[torch.tensor(0)], seq_lens=torch.tensor(0))
        assert len(results) == 2
        assert results[0].shape in [(1, 2), (1, 3), (1, 256)], results[0].shape
        assert results[1] == [torch.tensor(0)]
    elif tf_expected_to_work:
        model = tf.saved_model.load(os.path.join(export_dir, 'model'))
        assert model
        results = model(tf.convert_to_tensor(test_obs, dtype=tf.float32))
        assert len(results) == 2
        assert results[0].shape in [(1, 2), (1, 3), (1, 256)], results[0].shape
        assert results[1].shape == (1, 1), results[1].shape
    shutil.rmtree(export_dir)
    print('Exporting policy (`default_policy`) model ', alg_name, export_dir)
    if multi_agent:
        algo.export_policy_model(export_dir, policy_id='pol1')
        algo.export_policy_model(export_dir + '_2', policy_id='pol2')
    else:
        algo.export_policy_model(export_dir, policy_id=DEFAULT_POLICY_ID)
    if framework == 'torch':
        filename = os.path.join(export_dir, 'model.pt')
        model = torch.load(filename)
        assert model
        results = model(input_dict={'obs': torch.from_numpy(test_obs)}, state=[torch.tensor(0)], seq_lens=torch.tensor(0))
        assert len(results) == 2
        assert results[0].shape in [(1, 2), (1, 3), (1, 256)], results[0].shape
        assert results[1] == [torch.tensor(0)]
    elif tf_expected_to_work:
        model = tf.saved_model.load(export_dir)
        assert model
        results = model(tf.convert_to_tensor(test_obs, dtype=tf.float32))
        assert len(results) == 2
        assert results[0].shape in [(1, 2), (1, 3), (1, 256)], results[0].shape
        assert results[1].shape == (1, 1), results[1].shape
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
        if multi_agent:
            shutil.rmtree(export_dir + '_2')
    algo.stop()

class TestExportCheckpointAndModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            return 10
        ray.init()

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        ray.shutdown()

    def test_export_appo(self):
        if False:
            print('Hello World!')
        for fw in framework_iterator():
            export_test('APPO', fw)

    def test_export_ppo(self):
        if False:
            i = 10
            return i + 15
        for fw in framework_iterator():
            export_test('PPO', fw)

    def test_export_ppo_multi_agent(self):
        if False:
            while True:
                i = 10
        for fw in framework_iterator():
            export_test('PPO', fw, multi_agent=True)

    def test_export_sac(self):
        if False:
            return 10
        for fw in framework_iterator():
            export_test('SAC', fw, tf_expected_to_work=False)
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))