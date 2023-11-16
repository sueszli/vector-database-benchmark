import time
from random import random, uniform, gauss

def sleep(seconds, delta=0.3):
    if False:
        print('Hello World!')
    time.sleep(jitter(seconds, delta))

def jitter(value, delta=0.3):
    if False:
        for i in range(10):
            print('nop')
    jitter = delta * value
    return uniform(value - jitter, value + jitter)

def action_delay(low, high):
    if False:
        while True:
            i = 10
    longNum = uniform(low, high)
    shortNum = float('{0:.2f}'.format(longNum))
    time.sleep(shortNum)

def random_lat_long_delta():
    if False:
        for i in range(10):
            print('nop')
    return (random() * 1e-05 - 5e-06) * 5

def random_alt_delta():
    if False:
        for i in range(10):
            print('nop')
    return uniform(-0.2, 0.2)

def gps_noise_rng(radius):
    if False:
        i = 10
        return i + 15
    '\n    Simulates gps noise.\n    '
    noise = gauss(0, radius / 3.0)
    noise = min(max(-radius, noise), radius)
    return noise