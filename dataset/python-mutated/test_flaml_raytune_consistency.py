import numpy as np
import time
from flaml import tune

def evaluate_config(config):
    if False:
        i = 10
        return i + 15
    'evaluate a hyperparameter configuration'
    metric = (round(config['x']) - 85000) ** 2 - config['x'] / config['y']
    time.sleep(config['x'] / 100000)
    tune.report(metric=metric)
config_search_space = {'x': tune.lograndint(lower=1, upper=100000), 'y': tune.randint(lower=1, upper=100000)}
low_cost_partial_config = {'x': 1}

def setup_searcher(searcher_name):
    if False:
        return 10
    from flaml.tune.searcher.blendsearch import BlendSearch, CFO, RandomSearch
    if 'cfo' in searcher_name:
        searcher = CFO(space=config_search_space, low_cost_partial_config=low_cost_partial_config)
    elif searcher_name == 'bs':
        searcher = BlendSearch(metric='metric', mode='min', space=config_search_space, low_cost_partial_config=low_cost_partial_config)
    elif searcher_name == 'random':
        searcher = RandomSearch(space=config_search_space)
    else:
        return None
    return searcher

def _test_flaml_raytune_consistency(num_samples=-1, max_concurrent_trials=1, searcher_name='cfo'):
    if False:
        print('Hello World!')
    try:
        from ray import tune as raytune, __version__ as ray_version
        if ray_version.startswith('1.'):
            from ray.tune.suggest import ConcurrencyLimiter
        else:
            from ray.tune.search import ConcurrencyLimiter
    except ImportError:
        print('skip _test_flaml_raytune_consistency because ray tune cannot be imported.')
        return
    searcher = setup_searcher(searcher_name)
    analysis = tune.run(evaluate_config, config=config_search_space, low_cost_partial_config=low_cost_partial_config, metric='metric', mode='min', num_samples=num_samples, time_budget_s=None, local_dir='logs/', search_alg=searcher)
    flaml_best_config = analysis.best_config
    flaml_config_in_results = [v['config'] for v in analysis.results.values()]
    flaml_time_in_results = [v['time_total_s'] for v in analysis.results.values()]
    print(analysis.best_trial.last_result)
    searcher = setup_searcher(searcher_name)
    search_alg = ConcurrencyLimiter(searcher, max_concurrent_trials)
    analysis = raytune.run(evaluate_config, config=config_search_space, metric='metric', mode='min', num_samples=num_samples, local_dir='logs/', search_alg=search_alg)
    ray_best_config = analysis.best_config
    ray_config_in_results = [v['config'] for v in analysis.results.values()]
    ray_time_in_results = [v['time_total_s'] for v in analysis.results.values()]
    print(analysis.best_trial.last_result)
    print('time_total_s in flaml', flaml_time_in_results)
    print('time_total_s in ray', ray_time_in_results)
    print('best flaml', searcher_name, flaml_best_config)
    print('ray best', searcher_name, ray_best_config)
    print('flaml config in results', searcher_name, flaml_config_in_results)
    print('ray config in results', searcher_name, ray_config_in_results)
    assert ray_best_config == flaml_best_config, 'best config should be the same'
    assert flaml_config_in_results == ray_config_in_results, 'results from raytune and flaml should be the same'

def test_consistency():
    if False:
        for i in range(10):
            print('nop')
    _test_flaml_raytune_consistency(num_samples=5, max_concurrent_trials=1, searcher_name='random')
    _test_flaml_raytune_consistency(num_samples=5, max_concurrent_trials=1, searcher_name='cfo')
    _test_flaml_raytune_consistency(num_samples=5, max_concurrent_trials=1, searcher_name='bs')
if __name__ == '__main__':
    test_consistency()