from collections import defaultdict
import math
import sys
import time
from picard import log
from picard.webservice.utils import hostkey_from_url
REQUEST_DELAY_MINIMUM = defaultdict(lambda : 1000)
REQUEST_DELAY = defaultdict(lambda : 1000)
REQUEST_DELAY_EXPONENT = defaultdict(lambda : 0)
CONGESTION_UNACK = defaultdict(lambda : 0)
CONGESTION_WINDOW_SIZE = defaultdict(lambda : 1.0)
CONGESTION_SSTHRESH = defaultdict(lambda : 0)
LAST_REQUEST_TIMES = defaultdict(lambda : 0)

def set_minimum_delay(hostkey, delay_ms):
    if False:
        for i in range(10):
            print('nop')
    'Set the minimun delay between requests\n            hostkey is an unique key, for example (host, port)\n            delay_ms is the delay in milliseconds\n    '
    REQUEST_DELAY_MINIMUM[hostkey] = delay_ms

def set_minimum_delay_for_url(url, delay_ms):
    if False:
        return 10
    'Set the minimun delay between requests\n            url will be converted to an unique key (host, port)\n            delay_ms is the delay in milliseconds\n    '
    set_minimum_delay(hostkey_from_url(url), delay_ms)

def current_delay(hostkey):
    if False:
        i = 10
        return i + 15
    'Returns the current delay (adaptive) between requests for this hostkey\n            hostkey is an unique key, for example (host, port)\n    '
    return REQUEST_DELAY[hostkey]

def get_delay_to_next_request(hostkey):
    if False:
        for i in range(10):
            print('nop')
    'Calculate delay to next request to hostkey (host, port)\n       returns a tuple (wait, delay) where:\n           wait is True if a delay is needed\n           delay is the delay in milliseconds to next request\n    '
    if CONGESTION_UNACK[hostkey] >= int(CONGESTION_WINDOW_SIZE[hostkey]):
        return (True, sys.maxsize)
    interval = REQUEST_DELAY[hostkey]
    if not interval:
        log.debug('%s: Starting another request without delay', hostkey)
        return (False, 0)
    last_request = LAST_REQUEST_TIMES[hostkey]
    if not last_request:
        log.debug('%s: First request', hostkey)
        _remember_request_time(hostkey)
        return (False, interval)
    elapsed = (time.time() - last_request) * 1000
    if elapsed >= interval:
        log.debug('%s: Last request was %d ms ago, starting another one', hostkey, elapsed)
        return (False, interval)
    delay = int(math.ceil(interval - elapsed))
    log.debug('%s: Last request was %d ms ago, waiting %d ms before starting another one', hostkey, elapsed, delay)
    return (True, delay)

def _remember_request_time(hostkey):
    if False:
        for i in range(10):
            print('nop')
    if REQUEST_DELAY[hostkey]:
        LAST_REQUEST_TIMES[hostkey] = time.time()

def increment_requests(hostkey):
    if False:
        return 10
    'Store the request time for this hostkey, and increment counter\n       It has to be called on each request\n    '
    _remember_request_time(hostkey)
    CONGESTION_UNACK[hostkey] += 1
    log.debug('%s: Incrementing requests to: %d', hostkey, CONGESTION_UNACK[hostkey])

def decrement_requests(hostkey):
    if False:
        print('Hello World!')
    'Decrement counter, it has to be called on each reply\n    '
    assert CONGESTION_UNACK[hostkey] > 0
    CONGESTION_UNACK[hostkey] -= 1
    log.debug('%s: Decrementing requests to: %d', hostkey, CONGESTION_UNACK[hostkey])

def copy_minimal_delay(from_hostkey, to_hostkey):
    if False:
        while True:
            i = 10
    'Copy minimal delay from one hostkey to another\n        Useful for redirections\n    '
    if from_hostkey in REQUEST_DELAY_MINIMUM and to_hostkey not in REQUEST_DELAY_MINIMUM:
        REQUEST_DELAY_MINIMUM[to_hostkey] = REQUEST_DELAY_MINIMUM[from_hostkey]
        log.debug('%s: Copy minimun delay from %s, setting it to %dms', to_hostkey, from_hostkey, REQUEST_DELAY_MINIMUM[to_hostkey])

def adjust(hostkey, slow_down):
    if False:
        print('Hello World!')
    'Adjust `REQUEST` and `CONGESTION` metrics when a HTTP request completes.\n\n            Args:\n                    hostkey: `(host, port)`.\n                    slow_down: `True` if we encountered intermittent server trouble\n                    and need to slow down.\n    '
    if slow_down:
        _slow_down(hostkey)
    elif CONGESTION_UNACK[hostkey] <= CONGESTION_WINDOW_SIZE[hostkey]:
        _out_of_backoff(hostkey)

def _slow_down(hostkey):
    if False:
        return 10
    delay = max(pow(2, REQUEST_DELAY_EXPONENT[hostkey]) * 1000, REQUEST_DELAY_MINIMUM[hostkey])
    REQUEST_DELAY_EXPONENT[hostkey] = min(REQUEST_DELAY_EXPONENT[hostkey] + 1, 5)
    CONGESTION_SSTHRESH[hostkey] = int(CONGESTION_WINDOW_SIZE[hostkey] / 2.0)
    CONGESTION_WINDOW_SIZE[hostkey] = 1.0
    log.debug('%s: slowdown; delay: %dms -> %dms; ssthresh: %d; cws: %.3f', hostkey, REQUEST_DELAY[hostkey], delay, CONGESTION_SSTHRESH[hostkey], CONGESTION_WINDOW_SIZE[hostkey])
    REQUEST_DELAY[hostkey] = delay

def _out_of_backoff(hostkey):
    if False:
        print('Hello World!')
    REQUEST_DELAY_EXPONENT[hostkey] = 0
    delay = max(int(REQUEST_DELAY[hostkey] / 2), REQUEST_DELAY_MINIMUM[hostkey])
    cws = CONGESTION_WINDOW_SIZE[hostkey]
    sst = CONGESTION_SSTHRESH[hostkey]
    if sst and cws >= sst:
        phase = 'congestion avoidance'
        cws = cws + 1.0 / cws
    else:
        phase = 'slow start'
        cws += 1
    if REQUEST_DELAY[hostkey] != delay or CONGESTION_WINDOW_SIZE[hostkey] != cws:
        log.debug('%s: oobackoff; delay: %dms -> %dms; %s; window size %.3f -> %.3f', hostkey, REQUEST_DELAY[hostkey], delay, phase, CONGESTION_WINDOW_SIZE[hostkey], cws)
        CONGESTION_WINDOW_SIZE[hostkey] = cws
        REQUEST_DELAY[hostkey] = delay