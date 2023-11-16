"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf

[2] Mastering Atari with Discrete World Models - 2021
D. Hafner, T. Lillicrap, M. Norouzi, J. Ba
https://arxiv.org/pdf/2010.02193.pdf
"""
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config
from ray import tune
num_gpus = 0
config = DreamerV3Config()
w = config.world_model_lr
c = config.critic_lr

def _env_creator(ctx):
    if False:
        print('Hello World!')
    import flappy_bird_gymnasium
    import gymnasium as gym
    from supersuit.generic_wrappers import resize_v1
    from ray.rllib.algorithms.dreamerv3.utils.env_runner import NormalizedImageEnv
    return NormalizedImageEnv(resize_v1(gym.make('FlappyBird-rgb-v0', audio_on=False), x_size=64, y_size=64))
tune.register_env('flappy-bird', _env_creator)
config.environment('flappy-bird').resources(num_learner_workers=0 if num_gpus == 1 else num_gpus, num_gpus_per_learner_worker=1 if num_gpus else 0, num_cpus_for_local_worker=1).rollouts(num_envs_per_worker=8 * (num_gpus or 1), remote_worker_envs=True).reporting(metrics_num_episodes_for_smoothing=num_gpus or 1, report_images_and_videos=False, report_dream_data=False, report_individual_batch_item_stats=False).training(model_size='M', training_ratio=64, batch_size_B=16 * (num_gpus or 1), world_model_lr=[[0, 0.4 * w], [8000, 0.4 * w], [10000, 3 * w]], critic_lr=[[0, 0.4 * c], [8000, 0.4 * c], [10000, 3 * c]], actor_lr=[[0, 0.4 * c], [8000, 0.4 * c], [10000, 3 * c]])