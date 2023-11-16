"""This example demonstrates basic Ray Tune random search and grid search."""
import time
import ray
from ray import train, tune

def evaluation_fn(step, width, height):
    if False:
        print('Hello World!')
    time.sleep(0.1)
    return (0.1 + width * step / 100) ** (-1) + height * 0.1

def easy_objective(config):
    if False:
        print('Hello World!')
    (width, height) = (config['width'], config['height'])
    for step in range(config['steps']):
        intermediate_score = evaluation_fn(step, width, height)
        train.report({'iterations': step, 'mean_loss': intermediate_score})
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true', help='Finish quickly for testing')
    (args, _) = parser.parse_known_args()
    ray.init(configure_logging=False)
    tuner = tune.Tuner(easy_objective, tune_config=tune.TuneConfig(metric='mean_loss', mode='min', num_samples=5 if args.smoke_test else 50), param_space={'steps': 5 if args.smoke_test else 100, 'width': tune.uniform(0, 20), 'height': tune.uniform(-100, 100), 'activation': tune.grid_search(['relu', 'tanh'])})
    results = tuner.fit()
    print('Best hyperparameters found were: ', results.get_best_result().config)