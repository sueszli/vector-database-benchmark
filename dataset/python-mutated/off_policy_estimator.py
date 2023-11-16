import gymnasium as gym
import numpy as np
import tree
from typing import Dict, Any, List
import logging
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import convert_ma_batch_to_sample_batch
from ray.rllib.utils.policy import compute_log_likelihoods_from_input_dict
from ray.rllib.utils.annotations import DeveloperAPI, ExperimentalAPI, OverrideToImplementCustomLogic
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import TensorType, SampleBatchType
from ray.rllib.offline.offline_evaluator import OfflineEvaluator
logger = logging.getLogger(__name__)

@DeveloperAPI
class OffPolicyEstimator(OfflineEvaluator):
    """Interface for an off policy estimator for counterfactual evaluation."""

    @DeveloperAPI
    def __init__(self, policy: Policy, gamma: float=0.0, epsilon_greedy: float=0.0):
        if False:
            print('Hello World!')
        'Initializes an OffPolicyEstimator instance.\n\n        Args:\n            policy: Policy to evaluate.\n            gamma: Discount factor of the environment.\n            epsilon_greedy: The probability by which we act acording to a fully random\n            policy during deployment. With 1-epsilon_greedy we act according the target\n            policy.\n            # TODO (kourosh): convert the input parameters to a config dict.\n        '
        super().__init__(policy)
        self.gamma = gamma
        self.epsilon_greedy = epsilon_greedy

    @DeveloperAPI
    def estimate_on_single_episode(self, episode: SampleBatch) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Returns off-policy estimates for the given one episode.\n\n        Args:\n            batch: The episode to calculate the off-policy estimates (OPE) on. The\n            episode must be a sample batch type that contains the fields "obs",\n            "actions", and "action_prob" and it needs to represent a\n            complete trajectory.\n\n        Returns:\n            The off-policy estimates (OPE) calculated on the given episode. The returned\n            dict can be any arbitrary mapping of strings to metrics.\n        '
        raise NotImplementedError

    @DeveloperAPI
    def estimate_on_single_step_samples(self, batch: SampleBatch) -> Dict[str, List[float]]:
        if False:
            i = 10
            return i + 15
        'Returns off-policy estimates for the batch of single timesteps. This is\n        highly optimized for bandits assuming each episode is a single timestep.\n\n        Args:\n            batch: The batch to calculate the off-policy estimates (OPE) on. The\n            batch must be a sample batch type that contains the fields "obs",\n            "actions", and "action_prob".\n\n        Returns:\n            The off-policy estimates (OPE) calculated on the given batch of single time\n            step samples. The returned dict can be any arbitrary mapping of strings to\n            a list of floats capturing the values per each record.\n        '
        raise NotImplementedError

    def on_before_split_batch_by_episode(self, sample_batch: SampleBatch) -> SampleBatch:
        if False:
            while True:
                i = 10
        'Called before the batch is split by episode. You can perform any\n        preprocessing on the batch that you want here.\n        e.g. adding done flags to the batch, or reseting some stats that you want to\n        track per episode later during estimation, .etc.\n\n        Args:\n            sample_batch: The batch to split by episode. This contains multiple\n            episodes.\n\n        Returns:\n            The modified batch before calling split_by_episode().\n        '
        return sample_batch

    @OverrideToImplementCustomLogic
    def on_after_split_batch_by_episode(self, all_episodes: List[SampleBatch]) -> List[SampleBatch]:
        if False:
            while True:
                i = 10
        'Called after the batch is split by episode. You can perform any\n        postprocessing on each episode that you want here.\n        e.g. computing advantage per episode, .etc.\n\n        Args:\n            all_episodes: The list of episodes in the original batch. Each element is a\n            sample batch type that is a single episode.\n        '
        return all_episodes

    @OverrideToImplementCustomLogic
    def peek_on_single_episode(self, episode: SampleBatch) -> None:
        if False:
            return 10
        'This is called on each episode before it is passed to\n        estimate_on_single_episode(). Using this method, you can get a peek at the\n        entire validation dataset before runnining the estimation. For examlpe if you\n        need to perform any normalizations of any sorts on the dataset, you can compute\n        the normalization parameters here.\n\n        Args:\n            episode: The episode that is split from the original batch. This is a\n            sample batch type that is a single episode.\n        '
        pass

    @DeveloperAPI
    def estimate(self, batch: SampleBatchType, split_batch_by_episode: bool=True) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        'Compute off-policy estimates.\n\n        Args:\n            batch: The batch to calculate the off-policy estimates (OPE) on. The\n            batch must contain the fields "obs", "actions", and "action_prob".\n            split_batch_by_episode: Whether to split the batch by episode.\n\n        Returns:\n            The off-policy estimates (OPE) calculated on the given batch. The returned\n            dict can be any arbitrary mapping of strings to metrics.\n            The dict consists of the following metrics:\n            - v_behavior: The discounted return averaged over episodes in the batch\n            - v_behavior_std: The standard deviation corresponding to v_behavior\n            - v_target: The estimated discounted return for `self.policy`,\n            averaged over episodes in the batch\n            - v_target_std: The standard deviation corresponding to v_target\n            - v_gain: v_target / max(v_behavior, 1e-8)\n            - v_delta: The difference between v_target and v_behavior.\n        '
        batch = convert_ma_batch_to_sample_batch(batch)
        self.check_action_prob_in_batch(batch)
        estimates_per_epsiode = []
        if split_batch_by_episode:
            batch = self.on_before_split_batch_by_episode(batch)
            all_episodes = batch.split_by_episode()
            all_episodes = self.on_after_split_batch_by_episode(all_episodes)
            for episode in all_episodes:
                assert len(set(episode[SampleBatch.EPS_ID])) == 1, 'The episode must contain only one episode id. For some reason the split_by_episode() method could not successfully split the batch by episodes. Each row in the dataset should be one episode. Check your evaluation dataset for errors.'
                self.peek_on_single_episode(episode)
            for episode in all_episodes:
                estimate_step_results = self.estimate_on_single_episode(episode)
                estimates_per_epsiode.append(estimate_step_results)
            estimates_per_epsiode = tree.map_structure(lambda *x: list(x), *estimates_per_epsiode)
        else:
            estimates_per_epsiode = self.estimate_on_single_step_samples(batch)
        estimates = {'v_behavior': np.mean(estimates_per_epsiode['v_behavior']), 'v_behavior_std': np.std(estimates_per_epsiode['v_behavior']), 'v_target': np.mean(estimates_per_epsiode['v_target']), 'v_target_std': np.std(estimates_per_epsiode['v_target'])}
        estimates['v_gain'] = estimates['v_target'] / max(estimates['v_behavior'], 1e-08)
        estimates['v_delta'] = estimates['v_target'] - estimates['v_behavior']
        return estimates

    @DeveloperAPI
    def check_action_prob_in_batch(self, batch: SampleBatchType) -> None:
        if False:
            i = 10
            return i + 15
        'Checks if we support off policy estimation (OPE) on given batch.\n\n        Args:\n            batch: The batch to check.\n\n        Raises:\n            ValueError: In case `action_prob` key is not in batch\n        '
        if 'action_prob' not in batch:
            raise ValueError("Off-policy estimation is not possible unless the inputs include action probabilities (i.e., the policy is stochastic and emits the 'action_prob' key). For DQN this means using `exploration_config: {type: 'SoftQ'}`. You can also set `off_policy_estimation_methods: {}` to disable estimation.")

    @ExperimentalAPI
    def compute_action_probs(self, batch: SampleBatch):
        if False:
            return 10
        log_likelihoods = compute_log_likelihoods_from_input_dict(self.policy, batch)
        new_prob = np.exp(convert_to_numpy(log_likelihoods))
        if self.epsilon_greedy > 0.0:
            if not isinstance(self.policy.action_space, gym.spaces.Discrete):
                raise ValueError('Evaluation with epsilon-greedy exploration is only supported with discrete action spaces.')
            eps = self.epsilon_greedy
            new_prob = new_prob * (1 - eps) + eps / self.policy.action_space.n
        return new_prob

    @DeveloperAPI
    def train(self, batch: SampleBatchType) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        'Train a model for Off-Policy Estimation.\n\n        Args:\n            batch: SampleBatch to train on\n\n        Returns:\n            Any optional metrics to return from the estimator\n        '
        return {}

    @Deprecated(old='OffPolicyEstimator.action_log_likelihood', new='ray.rllib.utils.policy.compute_log_likelihoods_from_input_dict', error=True)
    def action_log_likelihood(self, batch: SampleBatchType) -> TensorType:
        if False:
            for i in range(10):
                print('nop')
        log_likelihoods = compute_log_likelihoods_from_input_dict(self.policy, batch)
        return convert_to_numpy(log_likelihoods)