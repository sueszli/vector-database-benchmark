import argparse
import gymnasium as gym
import shutil
import tempfile
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.ppo.tf.ppo_tf_rl_module import PPOTfRLModule
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec

def _parse_args():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', choices=['tf2', 'torch'], default='torch')
    return parser.parse_args()
if __name__ == '__main__':
    args = _parse_args()
    ray.init()
    module_class = PPOTfRLModule if args.framework == 'tf2' else PPOTorchRLModule
    env = gym.make('CartPole-v1')
    module_to_load = SingleAgentRLModuleSpec(module_class=module_class, model_config_dict={'fcnet_hiddens': [32]}, catalog_class=PPOCatalog, observation_space=env.observation_space, action_space=env.action_space).build()
    CHECKPOINT_DIR = tempfile.mkdtemp()
    module_to_load.save_to_checkpoint(CHECKPOINT_DIR)
    module_to_load_spec = SingleAgentRLModuleSpec(module_class=module_class, model_config_dict={'fcnet_hiddens': [32]}, catalog_class=PPOCatalog, load_state_path=CHECKPOINT_DIR)
    config = PPOConfig().experimental(_enable_new_api_stack=True).framework(args.framework).rl_module(rl_module_spec=module_to_load_spec).environment('CartPole-v1')
    tuner = tune.Tuner('PPO', param_space=config.to_dict(), run_config=air.RunConfig(stop={'training_iteration': 1}, failure_config=air.FailureConfig(fail_fast='raise')))
    tuner.fit()
    shutil.rmtree(CHECKPOINT_DIR)