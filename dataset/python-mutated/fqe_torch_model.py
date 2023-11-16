from typing import Dict, Any
from ray.rllib.models.utils import get_initializer
from ray.rllib.policy import Policy
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.annotations import is_overridden
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from gymnasium.spaces import Discrete
(torch, nn) = try_import_torch()

@DeveloperAPI
class FQETorchModel:
    """Pytorch implementation of the Fitted Q-Evaluation (FQE) model from
    https://arxiv.org/abs/1911.06854
    """

    def __init__(self, policy: Policy, gamma: float, model_config: ModelConfigDict=None, n_iters: int=1, lr: float=0.001, min_loss_threshold: float=0.0001, clip_grad_norm: float=100.0, minibatch_size: int=None, polyak_coef: float=1.0) -> None:
        if False:
            return 10
        '\n        Args:\n            policy: Policy to evaluate.\n            gamma: Discount factor of the environment.\n            model_config: The ModelConfigDict for self.q_model, defaults to:\n                {\n                    "fcnet_hiddens": [8, 8],\n                    "fcnet_activation": "relu",\n                    "vf_share_layers": True,\n                },\n            n_iters: Number of gradient steps to run on batch, defaults to 1\n            lr: Learning rate for Adam optimizer\n            min_loss_threshold: Early stopping if mean loss < min_loss_threshold\n            clip_grad_norm: Clip loss gradients to this maximum value\n            minibatch_size: Minibatch size for training Q-function;\n                if None, train on the whole batch\n            polyak_coef: Polyak averaging factor for target Q-function\n        '
        self.policy = policy
        assert isinstance(policy.action_space, Discrete), f'{self.__class__.__name__} only supports discrete action spaces!'
        self.gamma = gamma
        self.observation_space = policy.observation_space
        self.action_space = policy.action_space
        if model_config is None:
            model_config = {'fcnet_hiddens': [32, 32, 32], 'fcnet_activation': 'relu', 'vf_share_layers': True}
        self.model_config = model_config
        self.device = self.policy.device
        self.q_model: TorchModelV2 = ModelCatalog.get_model_v2(self.observation_space, self.action_space, self.action_space.n, model_config, framework='torch', name='TorchQModel').to(self.device)
        self.target_q_model: TorchModelV2 = ModelCatalog.get_model_v2(self.observation_space, self.action_space, self.action_space.n, model_config, framework='torch', name='TargetTorchQModel').to(self.device)
        self.n_iters = n_iters
        self.lr = lr
        self.min_loss_threshold = min_loss_threshold
        self.clip_grad_norm = clip_grad_norm
        self.minibatch_size = minibatch_size
        self.polyak_coef = polyak_coef
        self.optimizer = torch.optim.Adam(self.q_model.variables(), self.lr)
        initializer = get_initializer('xavier_uniform', framework='torch')
        self.update_target(polyak_coef=1.0)

        def f(m):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(m, nn.Linear):
                initializer(m.weight)
        self.initializer = f

    def train(self, batch: SampleBatch) -> TensorType:
        if False:
            return 10
        'Trains self.q_model using FQE loss on given batch.\n\n        Args:\n            batch: A SampleBatch of episodes to train on\n\n        Returns:\n            A list of losses for each training iteration\n        '
        losses = []
        minibatch_size = self.minibatch_size or batch.count
        batch = batch.copy(shallow=True)
        for _ in range(self.n_iters):
            minibatch_losses = []
            batch.shuffle()
            for idx in range(0, batch.count, minibatch_size):
                minibatch = batch[idx:idx + minibatch_size]
                obs = torch.tensor(minibatch[SampleBatch.OBS], device=self.device)
                actions = torch.tensor(minibatch[SampleBatch.ACTIONS], device=self.device, dtype=int)
                rewards = torch.tensor(minibatch[SampleBatch.REWARDS], device=self.device)
                next_obs = torch.tensor(minibatch[SampleBatch.NEXT_OBS], device=self.device)
                dones = torch.tensor(minibatch[SampleBatch.TERMINATEDS], device=self.device, dtype=float)
                (q_values, _) = self.q_model({'obs': obs}, [], None)
                q_acts = torch.gather(q_values, -1, actions.unsqueeze(-1)).squeeze(-1)
                next_action_probs = self._compute_action_probs(next_obs)
                with torch.no_grad():
                    (next_q_values, _) = self.target_q_model({'obs': next_obs}, [], None)
                next_v = torch.sum(next_q_values * next_action_probs, axis=-1)
                targets = rewards + (1 - dones) * self.gamma * next_v
                loss = (targets - q_acts) ** 2
                loss = torch.mean(loss)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad.clip_grad_norm_(self.q_model.variables(), self.clip_grad_norm)
                self.optimizer.step()
                minibatch_losses.append(loss.item())
            iter_loss = sum(minibatch_losses) / len(minibatch_losses)
            losses.append(iter_loss)
            if iter_loss < self.min_loss_threshold:
                break
            self.update_target()
        return losses

    def estimate_q(self, batch: SampleBatch) -> TensorType:
        if False:
            return 10
        obs = torch.tensor(batch[SampleBatch.OBS], device=self.device)
        with torch.no_grad():
            (q_values, _) = self.q_model({'obs': obs}, [], None)
        actions = torch.tensor(batch[SampleBatch.ACTIONS], device=self.device, dtype=int)
        q_values = torch.gather(q_values, -1, actions.unsqueeze(-1)).squeeze(-1)
        return q_values

    def estimate_v(self, batch: SampleBatch) -> TensorType:
        if False:
            print('Hello World!')
        obs = torch.tensor(batch[SampleBatch.OBS], device=self.device)
        with torch.no_grad():
            (q_values, _) = self.q_model({'obs': obs}, [], None)
        action_probs = self._compute_action_probs(obs)
        v_values = torch.sum(q_values * action_probs, axis=-1)
        return v_values

    def update_target(self, polyak_coef=None):
        if False:
            return 10
        polyak_coef = polyak_coef or self.polyak_coef
        model_state_dict = self.q_model.state_dict()
        target_state_dict = self.target_q_model.state_dict()
        model_state_dict = {k: polyak_coef * model_state_dict[k] + (1 - polyak_coef) * v for (k, v) in target_state_dict.items()}
        self.target_q_model.load_state_dict(model_state_dict)

    def _compute_action_probs(self, obs: TensorType) -> TensorType:
        if False:
            return 10
        'Compute action distribution over the action space.\n\n        Args:\n            obs: A tensor of observations of shape (batch_size * obs_dim)\n\n        Returns:\n            action_probs: A tensor of action probabilities\n            of shape (batch_size * action_dim)\n        '
        input_dict = {SampleBatch.OBS: obs}
        seq_lens = torch.ones(len(obs), device=self.device, dtype=int)
        state_batches = []
        if is_overridden(self.policy.action_distribution_fn):
            try:
                (dist_inputs, dist_class, _) = self.policy.action_distribution_fn(self.policy.model, obs_batch=input_dict, state_batches=state_batches, seq_lens=seq_lens, explore=False, is_training=False)
            except TypeError:
                (dist_inputs, dist_class, _) = self.policy.action_distribution_fn(self.policy, self.policy.model, input_dict=input_dict, state_batches=state_batches, seq_lens=seq_lens, explore=False, is_training=False)
        else:
            dist_class = self.policy.dist_class
            (dist_inputs, _) = self.policy.model(input_dict, state_batches, seq_lens)
        action_dist = dist_class(dist_inputs, self.policy.model)
        assert isinstance(action_dist.dist, torch.distributions.categorical.Categorical), 'FQE only supports Categorical or MultiCategorical distributions!'
        action_probs = action_dist.dist.probs
        return action_probs

    def get_state(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        'Returns the current state of the FQE Model.'
        return {'policy_state': self.policy.get_state(), 'model_config': self.model_config, 'n_iters': self.n_iters, 'lr': self.lr, 'min_loss_threshold': self.min_loss_threshold, 'clip_grad_norm': self.clip_grad_norm, 'minibatch_size': self.minibatch_size, 'polyak_coef': self.polyak_coef, 'gamma': self.gamma, 'q_model_state': self.q_model.state_dict(), 'target_q_model_state': self.target_q_model.state_dict()}

    def set_state(self, state: Dict[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Sets the current state of the FQE Model.\n        Args:\n            state: A state dict returned by `get_state()`.\n        '
        self.n_iters = state['n_iters']
        self.lr = state['lr']
        self.min_loss_threshold = state['min_loss_threshold']
        self.clip_grad_norm = state['clip_grad_norm']
        self.minibatch_size = state['minibatch_size']
        self.polyak_coef = state['polyak_coef']
        self.gamma = state['gamma']
        self.policy.set_state(state['policy_state'])
        self.q_model.load_state_dict(state['q_model_state'])
        self.target_q_model.load_state_dict(state['target_q_model_state'])

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> 'FQETorchModel':
        if False:
            i = 10
            return i + 15
        'Creates a FQE Model from a state dict.\n\n        Args:\n            state: A state dict returned by `get_state`.\n\n        Returns:\n            An instance of the FQETorchModel.\n        '
        policy = Policy.from_state(state['policy_state'])
        model = cls(policy=policy, gamma=state['gamma'], model_config=state['model_config'])
        model.set_state(state)
        return model