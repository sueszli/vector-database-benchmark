"""
This example shows how to pretrain an RLModule using behavioral cloning from offline
data and, thereafter training it online with PPO.
"""
import gymnasium as gym
import shutil
import tempfile
import torch
from typing import Mapping
import ray
from ray import tune
from ray.train import RunConfig, FailureConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.base import ACTOR, ENCODER_OUT
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
GYM_ENV_NAME = 'CartPole-v1'
GYM_ENV = gym.make(GYM_ENV_NAME)

class BCActor(torch.nn.Module):
    """A wrapper for the encoder and policy networks of a PPORLModule.

    Args:
        encoder_network: The encoder network of the PPORLModule.
        policy_network: The policy network of the PPORLModule.
        distribution_cls: The distribution class to construct with the logits outputed
            by the policy network.

    """

    def __init__(self, encoder_network: torch.nn.Module, policy_network: torch.nn.Module, distribution_cls: torch.distributions.Distribution):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.encoder_network = encoder_network
        self.policy_network = policy_network
        self.distribution_cls = distribution_cls

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.distributions.Distribution:
        if False:
            while True:
                i = 10
        'Return an action distribution output by the policy network.\n\n        batch: A dict containing the key "obs" mapping to a torch tensor of\n            observations.\n\n        '
        encoder_out = self.encoder_network(batch)[ENCODER_OUT][ACTOR]
        action_logits = self.policy_network(encoder_out)
        distribution = self.distribution_cls(logits=action_logits)
        return distribution

def train_ppo_module_with_bc_finetune(dataset: ray.data.Dataset, ppo_module_spec: SingleAgentRLModuleSpec) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Train an Actor with BC finetuning on dataset.\n\n    Args:\n        dataset: The dataset to train on.\n        module_spec: The module spec of the PPORLModule that will be trained\n            after its encoder and policy networks are pretrained with BC.\n\n    Returns:\n        The path to the checkpoint of the pretrained PPORLModule.\n    '
    batch_size = 512
    learning_rate = 0.001
    num_epochs = 10
    module = ppo_module_spec.build()
    BCActorNetwork = BCActor(module.encoder, module.pi, torch.distributions.Categorical)
    optim = torch.optim.Adam(BCActorNetwork.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for batch in dataset.iter_torch_batches(batch_size=batch_size, dtypes=torch.float32):
            action_dist = BCActorNetwork(batch)
            loss = -torch.mean(action_dist.log_prob(batch['actions']))
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f'Epoch {epoch} loss: {loss.detach().item()}')
    checkpoint_dir = tempfile.mkdtemp()
    module.save_to_checkpoint(checkpoint_dir)
    return checkpoint_dir

def train_ppo_agent_from_checkpointed_module(module_spec_from_ckpt: SingleAgentRLModuleSpec) -> float:
    if False:
        print('Hello World!')
    'Train a checkpointed RLModule using PPO.\n\n    Args:\n        module_spec_from_ckpt: The module spec of the checkpointed RLModule.\n\n    Returns:\n        The best reward mean achieved by the PPO agent.\n\n    '
    config = PPOConfig().experimental(_enable_new_api_stack=True).rl_module(rl_module_spec=module_spec_from_ckpt).environment(GYM_ENV_NAME).debugging(seed=0)
    tuner = tune.Tuner('PPO', param_space=config.to_dict(), run_config=RunConfig(stop={'training_iteration': 10}, failure_config=FailureConfig(fail_fast='raise'), verbose=2))
    results = tuner.fit()
    best_reward_mean = results.get_best_result().metrics['episode_reward_mean']
    return best_reward_mean
if __name__ == '__main__':
    ray.init()
    ray.data.set_progress_bars(False)
    ds = ray.data.read_json('s3://rllib-oss-tests/cartpole-expert')
    module_spec = SingleAgentRLModuleSpec(module_class=PPOTorchRLModule, observation_space=GYM_ENV.observation_space, action_space=GYM_ENV.action_space, model_config_dict={'fcnet_hiddens': [64, 64]}, catalog_class=PPOCatalog)
    module_checkpoint_path = train_ppo_module_with_bc_finetune(ds, module_spec)
    module_spec.load_state_path = module_checkpoint_path
    best_reward = train_ppo_agent_from_checkpointed_module(module_spec)
    assert best_reward > 300, 'The PPO agent with pretraining should achieve a reward of at least 300.'
    shutil.rmtree(module_checkpoint_path)