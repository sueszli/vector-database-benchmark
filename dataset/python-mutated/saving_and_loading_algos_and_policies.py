from ray.rllib.algorithms.ppo import PPOConfig
my_ppo_config = PPOConfig().environment('CartPole-v1')
my_ppo = my_ppo_config.build()
my_ppo.train()
save_result = my_ppo.save()
path_to_checkpoint = save_result.checkpoint.path
print(f"An Algorithm checkpoint has been created inside directory: '{path_to_checkpoint}'.")
my_ppo.stop()
from ray.rllib.algorithms.algorithm import Algorithm
my_new_ppo = Algorithm.from_checkpoint(path_to_checkpoint)
my_new_ppo.train()
my_new_ppo.stop()
my_new_ppo = my_ppo_config.build()
my_new_ppo.restore(save_result)
my_new_ppo.train()
my_new_ppo.stop()
import os
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
my_ma_config = PPOConfig().multi_agent(policies={'pol1', 'pol2'}, policy_mapping_fn=lambda agent_id, episode, worker, **kw: 'pol1' if agent_id == 'agent1' else 'pol2', policies_to_train=['pol1', 'pol2'])
my_ma_config.environment(MultiAgentCartPole, env_config={'num_agents': 2})
my_ma_algo = my_ma_config.build()
my_ma_algo.train()
ma_checkpoint_dir = my_ma_algo.save().checkpoint.path
print(f"An Algorithm checkpoint has been created inside directory: '{ma_checkpoint_dir}'.\nIndividual Policy checkpoints can be found in '{os.path.join(ma_checkpoint_dir, 'policies')}'.")
my_ma_algo_clone = Algorithm.from_checkpoint(ma_checkpoint_dir)
my_ma_algo_clone.stop()
my_ma_algo_only_pol1 = Algorithm.from_checkpoint(ma_checkpoint_dir, policy_ids=['pol1'], policy_mapping_fn=lambda agent_id, episode, worker, **kw: 'pol1', policies_to_train=['pol1'])
assert my_ma_algo_only_pol1.get_policy('pol2') is None
my_ma_algo_only_pol1.train()
my_ma_algo_only_pol1.stop()
policy1 = my_ma_algo.get_policy(policy_id='pol1')
policy1.export_checkpoint('/tmp/my_policy_checkpoint')
import numpy as np
from ray.rllib.policy.policy import Policy
my_restored_policy = Policy.from_checkpoint('/tmp/my_policy_checkpoint')
obs = np.array([0.0, 0.1, 0.2, 0.3])
action = my_restored_policy.compute_single_action(obs)
print(f'Computed action {action} from given CartPole observation.')
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
algo_w_5_policies = PPOConfig().environment(env=MultiAgentCartPole, env_config={'num_agents': 5}).multi_agent(policies={'pol0', 'pol1', 'pol2', 'pol3', 'pol4'}, policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: f'pol{agent_id}').build()
algo_w_5_policies.train()
path_to_checkpoint = algo_w_5_policies.save().checkpoint.path
print(f"An Algorithm checkpoint has been created inside directory: '{path_to_checkpoint}'. It should contain 5 policies in the 'policies/' sub dir.")
algo_w_5_policies.stop()

def new_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return 'pol0' if agent_id in ['agent0', 'agent1'] else 'pol1'
algo_w_2_policies = Algorithm.from_checkpoint(checkpoint=path_to_checkpoint, policy_ids={'pol0', 'pol1'}, policy_mapping_fn=new_policy_mapping_fn)
algo_w_2_policies.train()
algo_w_2_policies.stop()
from ray.rllib.algorithms.ppo import PPOConfig
ppo_config = PPOConfig().environment('Pendulum-v1').checkpointing(export_native_model_files=True)
ppo = ppo_config.build()
ppo.train()
ppo_policy = ppo.get_policy()
ppo_policy.export_model('/tmp/my_nn_model')
checkpoint_dir = ppo_policy.export_checkpoint('tmp/ppo_policy')
checkpoint_dir = ppo.save().checkpoint.path
ppo_policy.export_model('/tmp/my_nn_model', onnx=False)