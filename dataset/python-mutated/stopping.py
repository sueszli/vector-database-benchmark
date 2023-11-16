from ray import train

def my_trainable(config):
    if False:
        return 10
    i = 1
    while True:
        time.sleep(1)
        train.report({'mean_accuracy': min(i / 10, 1.0)})
        i += 1

def my_trainable(config):
    if False:
        return 10
    i = 1
    while True:
        train.report({'mean_accuracy': min(i / 10, 1.0)})
        i += 1
from ray import train, tune
tuner = tune.Tuner(my_trainable, run_config=train.RunConfig(stop={'training_iteration': 10, 'mean_accuracy': 0.8}))
result_grid = tuner.fit()
final_iter = result_grid[0].metrics['training_iteration']
assert final_iter == 8, final_iter
from ray import train, tune

def stop_fn(trial_id: str, result: dict) -> bool:
    if False:
        while True:
            i = 10
    return result['mean_accuracy'] >= 0.8 or result['training_iteration'] >= 10
tuner = tune.Tuner(my_trainable, run_config=train.RunConfig(stop=stop_fn))
result_grid = tuner.fit()
final_iter = result_grid[0].metrics['training_iteration']
assert final_iter == 8, final_iter
from ray import train, tune
from ray.tune import Stopper

class CustomStopper(Stopper):

    def __init__(self):
        if False:
            print('Hello World!')
        self.should_stop = False

    def __call__(self, trial_id: str, result: dict) -> bool:
        if False:
            i = 10
            return i + 15
        if not self.should_stop and result['mean_accuracy'] >= 0.8:
            self.should_stop = True
        return self.should_stop

    def stop_all(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Returns whether to stop trials and prevent new ones from starting.'
        return self.should_stop
stopper = CustomStopper()
tuner = tune.Tuner(my_trainable, run_config=train.RunConfig(stop=stopper), tune_config=tune.TuneConfig(num_samples=2))
result_grid = tuner.fit()
for result in result_grid:
    final_iter = result.metrics.get('training_iteration', 0)
    assert final_iter <= 8, final_iter
from ray import train, tune
import time

def my_failing_trainable(config):
    if False:
        while True:
            i = 10
    if config['should_fail']:
        raise RuntimeError('Failing (on purpose)!')
    time.sleep(10)
    train.report({'mean_accuracy': 0.9})
tuner = tune.Tuner(my_failing_trainable, param_space={'should_fail': tune.grid_search([True, False])}, run_config=train.RunConfig(failure_config=train.FailureConfig(fail_fast=True)))
result_grid = tuner.fit()
for result in result_grid:
    final_iter = result.metrics.get('training_iteration')
    assert not final_iter, final_iter
from ray import train, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
scheduler = AsyncHyperBandScheduler(time_attr='training_iteration')
tuner = tune.Tuner(my_trainable, run_config=train.RunConfig(stop={'training_iteration': 10}), tune_config=tune.TuneConfig(scheduler=scheduler, num_samples=2, metric='mean_accuracy', mode='max'))
result_grid = tuner.fit()

def my_trainable(config):
    if False:
        i = 10
        return i + 15
    i = 1
    while True:
        time.sleep(1)
        train.report({'mean_accuracy': min(i / 10, 1.0)})
        i += 1
from ray import train, tune
tuner = tune.Tuner(my_trainable, run_config=train.RunConfig(stop={'time_total_s': 5}))
result_grid = tuner.fit()
assert result_grid[0].metrics['training_iteration'] < 8
from ray import tune
tuner = tune.Tuner(my_trainable, tune_config=tune.TuneConfig(time_budget_s=5.0))
result_grid = tuner.fit()
assert result_grid[0].metrics['training_iteration'] < 8