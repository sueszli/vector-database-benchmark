"""Helper module to handle time related stuff"""
from time import sleep as original_sleep
from datetime import datetime
from random import gauss
from random import uniform
STDEV = 0.5
sleep_percentage = 1
sleep_percentage = sleep_percentage * uniform(0.9, 1.1)

def randomize_time(mean):
    if False:
        print('Hello World!')
    allowed_range = mean * STDEV
    stdev = allowed_range / 3
    t = 0
    while abs(mean - t) > allowed_range:
        t = gauss(mean, stdev)
    return t

def set_sleep_percentage(percentage):
    if False:
        return 10
    global sleep_percentage
    sleep_percentage = percentage / 100
    sleep_percentage = sleep_percentage * uniform(0.9, 1.1)

def sleep(t, custom_percentage=None):
    if False:
        while True:
            i = 10
    if custom_percentage is None:
        custom_percentage = sleep_percentage
    time = randomize_time(t) * custom_percentage
    original_sleep(time)

def sleep_actual(t):
    if False:
        i = 10
        return i + 15
    original_sleep(t)

def get_time(labels):
    if False:
        i = 10
        return i + 15
    'To get a use out of this helpful function\n    catch in the same order of passed parameters'
    if not isinstance(labels, list):
        labels = [labels]
    results = []
    for label in labels:
        if label == 'this_minute':
            results.append(datetime.now().strftime('%M'))
        if label == 'this_hour':
            results.append(datetime.now().strftime('%H'))
        elif label == 'today':
            results.append(datetime.now().strftime('%Y-%m-%d'))
    results = results if len(results) > 1 else results[0]
    return results