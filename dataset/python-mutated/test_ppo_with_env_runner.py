import unittest
import ray
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.algorithms.ppo.ppo_learner import LEARNER_RESULTS_CURR_ENTROPY_COEFF_KEY
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.learner.learner import LEARNER_RESULTS_CURR_LR_KEY
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.metrics.learner_info import LEARNER_INFO
from ray.rllib.utils.test_utils import check, check_train_results, framework_iterator

def get_model_config(framework, lstm=False):
    if False:
        i = 10
        return i + 15
    return dict(use_lstm=True, lstm_use_prev_action=True, lstm_use_prev_reward=True, lstm_cell_size=10, max_seq_len=20) if lstm else {'use_lstm': False}

class MyCallbacks(DefaultCallbacks):

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        if False:
            print('Hello World!')
        stats = result['info'][LEARNER_INFO][DEFAULT_POLICY_ID]
        check(stats[LEARNER_RESULTS_CURR_ENTROPY_COEFF_KEY], 0.05 if algorithm.iteration == 1 else 0.0)
        check(stats[LEARNER_RESULTS_CURR_LR_KEY], 7.5e-06 if algorithm.iteration == 1 else 5e-06)
        optim = algorithm.learner_group._learner.get_optimizer()
        actual_optimizer_lr = optim.param_groups[0]['lr'] if algorithm.config.framework_str == 'torch' else optim.lr
        check(stats[LEARNER_RESULTS_CURR_LR_KEY], actual_optimizer_lr)

class TestPPO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        ray.init()

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        ray.shutdown()

    def test_ppo_compilation_and_schedule_mixins(self):
        if False:
            for i in range(10):
                print('nop')
        'Test whether PPO can be built with all frameworks.'
        config = ppo.PPOConfig().experimental(_enable_new_api_stack=True).rollouts(env_runner_cls=SingleAgentEnvRunner, num_rollout_workers=0).training(num_sgd_iter=2, lr=[[0, 1e-05], [512, 0.0]], entropy_coeff=[[0, 0.1], [256, 0.0]], train_batch_size=128).callbacks(MyCallbacks).evaluation(evaluation_num_workers=2, evaluation_duration=3, evaluation_duration_unit='episodes', enable_async_evaluation=True)
        num_iterations = 2
        for fw in framework_iterator(config, frameworks=('torch', 'tf2')):
            for env in ['CartPole-v1', 'Pendulum-v1']:
                print('Env={}'.format(env))
                for lstm in [False]:
                    print('LSTM={}'.format(lstm))
                    config.training(model=get_model_config(fw, lstm=lstm)).framework(eager_tracing=False)
                    algo = config.build(env=env)
                    learner = algo.learner_group._learner
                    optim = learner.get_optimizer()
                    lr = optim.param_groups[0]['lr'] if fw == 'torch' else optim.lr
                    check(lr, config.lr[0][1])
                    entropy_coeff = learner.entropy_coeff_schedulers_per_module[DEFAULT_POLICY_ID].get_current_value()
                    check(entropy_coeff, 0.1)
                    for i in range(num_iterations):
                        results = algo.train()
                        check_train_results(results)
                        print(results)
                    algo.stop()
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))