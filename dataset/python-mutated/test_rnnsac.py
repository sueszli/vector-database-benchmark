import unittest
import ray
from ray.rllib.algorithms import sac
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_compute_single_action, framework_iterator
(tf1, tf, tfv) = try_import_tf()
(torch, nn) = try_import_torch()

class TestRNNSAC(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            print('Hello World!')
        ray.init()

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            return 10
        ray.shutdown()

    def test_rnnsac_compilation(self):
        if False:
            i = 10
            return i + 15
        'Test whether RNNSAC can be built on all frameworks.'
        config = sac.RNNSACConfig().environment('CartPole-v1').rollouts(num_rollout_workers=0).training(model={'max_seq_len': 20}, policy_model_config={'use_lstm': True, 'lstm_cell_size': 64, 'fcnet_hiddens': [10], 'lstm_use_prev_action': True, 'lstm_use_prev_reward': True}, q_model_config={'use_lstm': True, 'lstm_cell_size': 64, 'fcnet_hiddens': [10], 'lstm_use_prev_action': True, 'lstm_use_prev_reward': True}, replay_buffer_config={'type': 'MultiAgentPrioritizedReplayBuffer', 'replay_burn_in': 20, 'zero_init_states': True}, lr=0.0005, num_steps_sampled_before_learning_starts=0)
        num_iterations = 1
        for _ in framework_iterator(config, frameworks='torch'):
            algo = config.build()
            for i in range(num_iterations):
                results = algo.train()
                print(results)
            check_compute_single_action(algo, include_state=True, include_prev_action_reward=True)
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))