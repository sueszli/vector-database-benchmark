import numpy as np
import os
import ray
import time
ray.init(ignore_reinit_error=True)

@ray.remote(max_retries=1)
def potentially_fail(failure_probability):
    if False:
        i = 10
        return i + 15
    time.sleep(0.2)
    if np.random.random() < failure_probability:
        os._exit(0)
    return 0
for _ in range(3):
    try:
        ray.get(potentially_fail.remote(0.5))
        print('SUCCESS')
    except ray.exceptions.WorkerCrashedError:
        print('FAILURE')
import numpy as np
import os
import ray
import time
ray.init(ignore_reinit_error=True)

class RandomError(Exception):
    pass

@ray.remote(max_retries=1, retry_exceptions=True)
def potentially_fail(failure_probability):
    if False:
        for i in range(10):
            print('nop')
    if failure_probability < 0 or failure_probability > 1:
        raise ValueError(f'failure_probability must be between 0 and 1, but got: {failure_probability}')
    time.sleep(0.2)
    if np.random.random() < failure_probability:
        raise RandomError('Failed!')
    return 0
for _ in range(3):
    try:
        ray.get(potentially_fail.remote(0.5))
        print('SUCCESS')
    except RandomError:
        print('FAILURE')
retry_on_exception = potentially_fail.options(retry_exceptions=[RandomError])
try:
    ray.get(retry_on_exception.remote(-1))
except ValueError:
    print('FAILED AS EXPECTED')
else:
    raise RuntimeError("An exception should be raised so this shouldn't be reached.")
for _ in range(3):
    try:
        ray.get(retry_on_exception.remote(0.5))
        print('SUCCESS')
    except RandomError:
        print('FAILURE AFTER RETRIES')