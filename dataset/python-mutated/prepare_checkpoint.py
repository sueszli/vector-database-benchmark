import random
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.sac import SACConfig

def create_appo_cartpole_checkpoint(output_dir):
    if False:
        while True:
            i = 10
    config = APPOConfig().environment('CartPole-v1').rollouts(enable_connectors=True)
    algo = config.build()
    algo.save(checkpoint_dir=output_dir)

def create_open_spiel_checkpoint(output_dir):
    if False:
        for i in range(10):
            print('nop')

    def _policy_mapping_fn(*args, **kwargs):
        if False:
            while True:
                i = 10
        random.choice(['main', 'opponent'])
    config = SACConfig().environment('open_spiel_env').framework('tf2').rollouts(num_rollout_workers=1, num_envs_per_worker=5, enable_tf1_exec_eagerly=True).training(model={'fcnet_hiddens': [512, 512]}).multi_agent(policies={'main', 'opponent'}, policy_mapping_fn=_policy_mapping_fn, policies_to_train=['main'])
    algo = config.build()
    algo.save(checkpoint_dir=output_dir)