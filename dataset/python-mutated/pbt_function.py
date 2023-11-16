import argparse
import json
import os
import random
import tempfile
import numpy as np
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import PopulationBasedTraining

def pbt_function(config):
    if False:
        for i in range(10):
            print('nop')
    "Toy PBT problem for benchmarking adaptive learning rate.\n\n    The goal is to optimize this trainable's accuracy. The accuracy increases\n    fastest at the optimal lr, which is a function of the current accuracy.\n\n    The optimal lr schedule for this problem is the triangle wave as follows.\n    Note that many lr schedules for real models also follow this shape:\n\n     best lr\n      ^\n      |    /      |   /        |  /          | /            ------------> accuracy\n\n    In this problem, using PBT with a population of 2-4 is sufficient to\n    roughly approximate this lr schedule. Higher population sizes will yield\n    faster convergence. Training will not converge without PBT.\n    "
    lr = config['lr']
    checkpoint_interval = config.get('checkpoint_interval', 1)
    accuracy = 0.0
    step = 1
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            with open(os.path.join(checkpoint_dir, 'checkpoint.json'), 'r') as f:
                checkpoint_dict = json.load(f)
        accuracy = checkpoint_dict['acc']
        last_step = checkpoint_dict['step']
        step = last_step + 1
    midpoint = 100
    q_tolerance = 3
    noise_level = 2
    while True:
        if accuracy < midpoint:
            optimal_lr = 0.01 * accuracy / midpoint
        else:
            optimal_lr = 0.01 - 0.01 * (accuracy - midpoint) / midpoint
        optimal_lr = min(0.01, max(0.001, optimal_lr))
        q_err = max(lr, optimal_lr) / min(lr, optimal_lr)
        if q_err < q_tolerance:
            accuracy += 1.0 / q_err * random.random()
        elif lr > optimal_lr:
            accuracy -= (q_err - q_tolerance) * random.random()
        accuracy += noise_level * np.random.normal()
        accuracy = max(0, accuracy)
        metrics = {'mean_accuracy': accuracy, 'cur_lr': lr, 'optimal_lr': optimal_lr, 'q_err': q_err, 'done': accuracy > midpoint * 2}
        if step % checkpoint_interval == 0:
            with tempfile.TemporaryDirectory() as tempdir:
                with open(os.path.join(tempdir, 'checkpoint.json'), 'w') as f:
                    checkpoint_dict = {'acc': accuracy, 'step': step}
                    json.dump(checkpoint_dict, f)
                train.report(metrics, checkpoint=Checkpoint.from_directory(tempdir))
        else:
            train.report(metrics)
        step += 1

def run_tune_pbt(smoke_test=False):
    if False:
        i = 10
        return i + 15
    perturbation_interval = 5
    pbt = PopulationBasedTraining(time_attr='training_iteration', perturbation_interval=perturbation_interval, hyperparam_mutations={'lr': tune.uniform(0.0001, 0.02), 'some_other_factor': [1, 2]})
    tuner = tune.Tuner(pbt_function, run_config=train.RunConfig(name='pbt_function_api_example', verbose=False, stop={'done': True, 'training_iteration': 10 if smoke_test else 1000}, failure_config=train.FailureConfig(fail_fast=True), checkpoint_config=train.CheckpointConfig(checkpoint_score_attribute='mean_accuracy', num_to_keep=2)), tune_config=tune.TuneConfig(scheduler=pbt, metric='mean_accuracy', mode='max', num_samples=8), param_space={'lr': 0.0001, 'some_other_factor': 1, 'checkpoint_interval': perturbation_interval})
    results = tuner.fit()
    print('Best hyperparameters found were: ', results.get_best_result().config)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true', default=False, help='Finish quickly for testing')
    (args, _) = parser.parse_known_args()
    if args.smoke_test:
        ray.init(num_cpus=2)
    run_tune_pbt(smoke_test=args.smoke_test)