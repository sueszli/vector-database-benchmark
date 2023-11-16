"""Example of using PBT with RLlib.

Note that this requires a cluster with at least 8 GPUs in order for all trials
to run concurrently, otherwise PBT will round-robin train the trials which
is less efficient (or you can set {"gpu": 0} to use CPUs for SGD instead).

Note that Tune in general does not need 8 GPUs, and this is just a more
computationally demanding example.
"""
import random
from ray import train, tune
from ray.rllib.algorithms.ppo import PPO
from ray.tune.schedulers import PopulationBasedTraining
if __name__ == '__main__':

    def explore(config):
        if False:
            for i in range(10):
                print('nop')
        if config['train_batch_size'] < config['sgd_minibatch_size'] * 2:
            config['train_batch_size'] = config['sgd_minibatch_size'] * 2
        if config['num_sgd_iter'] < 1:
            config['num_sgd_iter'] = 1
        return config
    pbt = PopulationBasedTraining(time_attr='time_total_s', perturbation_interval=120, resample_probability=0.25, hyperparam_mutations={'lambda': lambda : random.uniform(0.9, 1.0), 'clip_param': lambda : random.uniform(0.01, 0.5), 'lr': [0.001, 0.0005, 0.0001, 5e-05, 1e-05], 'num_sgd_iter': lambda : random.randint(1, 30), 'sgd_minibatch_size': lambda : random.randint(128, 16384), 'train_batch_size': lambda : random.randint(2000, 160000)}, custom_explore_fn=explore)
    tuner = tune.Tuner(PPO, run_config=train.RunConfig(name='pbt_humanoid_test'), tune_config=tune.TuneConfig(scheduler=pbt, num_samples=8, metric='episode_reward_mean', mode='max'), param_space={'env': 'Humanoid-v1', 'kl_coeff': 1.0, 'num_workers': 8, 'num_gpus': 1, 'model': {'free_log_std': True}, 'lambda': 0.95, 'clip_param': 0.2, 'lr': 0.0001, 'num_sgd_iter': tune.choice([10, 20, 30]), 'sgd_minibatch_size': tune.choice([128, 512, 2048]), 'train_batch_size': tune.choice([10000, 20000, 40000])})
    results = tuner.fit()
    print('best hyperparameters: ', results.get_best_result().config)