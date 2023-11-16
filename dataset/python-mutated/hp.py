import ray.tune as tune
import numpy as np
from typing import List, Union, Callable

def uniform(lower: float, upper: float) -> 'tune.sample.Float':
    if False:
        while True:
            i = 10
    '\n    Sample a float uniformly between lower and upper.\n\n    :param lower: Lower bound of the sampling range.\n    :param upper: Upper bound of the sampling range.\n    '
    return tune.uniform(lower, upper)

def quniform(lower: float, upper: float, q: float) -> 'tune.sample.Float':
    if False:
        while True:
            i = 10
    '\n    Sample a float uniformly between lower and upper.\n    Round the result to nearest value with granularity q, include upper.\n\n    :param lower: Lower bound of the sampling range.\n    :param upper: Upper bound of the sampling range.\n    :param q: Granularity for increment.\n    '
    return tune.quniform(lower, upper, q)

def loguniform(lower: float, upper: float, base: int=10) -> 'tune.sample.Float':
    if False:
        for i in range(10):
            print('nop')
    '\n    Sample a float between lower and upper.\n    Power distribute uniformly between log_{base}(lower) and log_{base}(upper).\n\n    :param lower: Lower bound of the sampling range.\n    :param upper: Upper bound of the sampling range.\n    :param base: Log base for distribution. Default to 10.\n    '
    return tune.loguniform(lower, upper, base)

def qloguniform(lower: float, upper: float, q: float, base: int=10) -> 'tune.sample.Float':
    if False:
        while True:
            i = 10
    '\n    Sample a float between lower and upper.\n    Power distribute uniformly between log_{base}(lower) and log_{base}(upper).\n    Round the result to nearest value with granularity q, include upper.\n\n    :param lower: Lower bound of the sampling range.\n    :param upper: Upper bound of the sampling range.\n    :param q: Granularity for increment.\n    :param base: Log base for distribution. Default to 10.\n    '
    return tune.qloguniform(lower, upper, q, base)

def randn(mean: float=0.0, std: float=1.0) -> 'tune.sample.Float':
    if False:
        return 10
    '\n    Sample a float from normal distribution.\n\n    :param mean: Mean of the normal distribution. Default to 0.0.\n    :param std: Std of the normal distribution. Default to 1.0.\n    '
    return tune.randn(mean, std)

def qrandn(mean: float, std: float, q: float) -> 'tune.sample.Float':
    if False:
        for i in range(10):
            print('nop')
    '\n    Sample a float from normal distribution.\n    Round the result to nearest value with granularity q.\n\n    :param mean: Mean of the normal distribution. Default to 0.0.\n    :param std: Std of the normal distribution. Default to 1.0.\n    :param q: Granularity for increment.\n    '
    return tune.qrandn(mean, std, q)

def randint(lower: int, upper: int) -> 'tune.sample.Integer':
    if False:
        return 10
    '\n    Uniformly sample integer between lower and upper. (Both inclusive)\n\n    :param lower: Lower bound of the sampling range.\n    :param upper: Upper bound of the sampling range.\n    '
    return tune.randint(lower, upper)

def qrandint(lower: int, upper: int, q: int=1) -> 'tune.sample.Integer':
    if False:
        print('Hello World!')
    '\n    Uniformly sample integer between lower and upper. (Both inclusive)\n    Round the result to nearest value with granularity q.\n\n    :param lower: Lower bound of the sampling range.\n    :param upper: Upper bound of the sampling range.\n    :param q: Integer Granularity for increment.\n    '
    return tune.qrandint(lower, upper, q)

def choice(categories: List) -> 'tune.sample.Categorical':
    if False:
        i = 10
        return i + 15
    '\n    Uniformly sample from a list\n\n    :param categories: A list to be sampled.\n    '
    return tune.choice(categories)

def choice_n(categories: List, min_items: int, max_items: int) -> 'tune.sample.Function':
    if False:
        while True:
            i = 10
    '\n    Sample a subset from a list\n\n    :param categories: A list to be sampled\n    :param min_items: minimum number of items to be sampled\n    :param max_items: maximum number of items to be sampled\n    '
    return tune.sample_from(lambda spec: list(np.random.choice(categories, size=np.random.randint(low=min_items, high=max_items), replace=False)))

def sample_from(func: Callable) -> Callable:
    if False:
        print('Hello World!')
    '\n    Sample from a function.\n\n    :param func: The function to be sampled.\n    '
    return tune.sample_from(func)

def grid_search(values: List) -> dict:
    if False:
        return 10
    '\n    Specifying grid search over a list.\n\n    :param values: A list to be grid searched.\n    '
    return tune.grid_search(values)