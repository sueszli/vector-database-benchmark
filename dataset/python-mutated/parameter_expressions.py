"""
parameter_expression.py
"""
import numpy as np

def choice(options, random_state):
    if False:
        return 10
    '\n    options: 1-D array-like or int\n    random_state: an object of numpy.random.RandomState\n    '
    return random_state.choice(options)

def randint(lower, upper, random_state):
    if False:
        return 10
    '\n    Generate a random integer from `lower` (inclusive) to `upper` (exclusive).\n    lower: an int that represent an lower bound\n    upper: an int that represent an upper bound\n    random_state: an object of numpy.random.RandomState\n    '
    return random_state.randint(lower, upper)

def uniform(low, high, random_state):
    if False:
        for i in range(10):
            print('nop')
    '\n    low: an float that represent an lower bound\n    high: an float that represent an upper bound\n    random_state: an object of numpy.random.RandomState\n    '
    assert high >= low, 'Upper bound must be larger than lower bound'
    return random_state.uniform(low, high)

def quniform(low, high, q, random_state):
    if False:
        while True:
            i = 10
    '\n    low: an float that represent an lower bound\n    high: an float that represent an upper bound\n    q: sample step\n    random_state: an object of numpy.random.RandomState\n    '
    return np.clip(np.round(uniform(low, high, random_state) / q) * q, low, high)

def loguniform(low, high, random_state):
    if False:
        while True:
            i = 10
    '\n    low: an float that represent an lower bound\n    high: an float that represent an upper bound\n    random_state: an object of numpy.random.RandomState\n    '
    assert low > 0, 'Lower bound must be positive'
    return np.exp(uniform(np.log(low), np.log(high), random_state))

def qloguniform(low, high, q, random_state):
    if False:
        return 10
    '\n    low: an float that represent an lower bound\n    high: an float that represent an upper bound\n    q: sample step\n    random_state: an object of numpy.random.RandomState\n    '
    return np.clip(np.round(loguniform(low, high, random_state) / q) * q, low, high)

def normal(mu, sigma, random_state):
    if False:
        i = 10
        return i + 15
    '\n    The probability density function of the normal distribution,\n    first derived by De Moivre and 200 years later by both Gauss and Laplace independently.\n    mu: float or array_like of floats\n        Mean (“centre”) of the distribution.\n    sigma: float or array_like of floats\n           Standard deviation (spread or “width”) of the distribution.\n    random_state: an object of numpy.random.RandomState\n    '
    return random_state.normal(mu, sigma)

def qnormal(mu, sigma, q, random_state):
    if False:
        i = 10
        return i + 15
    '\n    mu: float or array_like of floats\n    sigma: float or array_like of floats\n    q: sample step\n    random_state: an object of numpy.random.RandomState\n    '
    return np.round(normal(mu, sigma, random_state) / q) * q

def lognormal(mu, sigma, random_state):
    if False:
        for i in range(10):
            print('nop')
    '\n    mu: float or array_like of floats\n    sigma: float or array_like of floats\n    random_state: an object of numpy.random.RandomState\n    '
    return np.exp(normal(mu, sigma, random_state))

def qlognormal(mu, sigma, q, random_state):
    if False:
        while True:
            i = 10
    '\n    mu: float or array_like of floats\n    sigma: float or array_like of floats\n    q: sample step\n    random_state: an object of numpy.random.RandomState\n    '
    return np.round(lognormal(mu, sigma, random_state) / q) * q