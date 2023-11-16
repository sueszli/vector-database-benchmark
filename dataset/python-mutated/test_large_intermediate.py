import time
import pytest
from ray.tests.conftest import *
import numpy as np
import ray
from ray import workflow

def test_simple_large_intermediate(workflow_start_regular_shared):
    if False:
        while True:
            i = 10

    @ray.remote
    def large_input():
        if False:
            i = 10
            return i + 15
        return np.arange(2 ** 24)

    @ray.remote
    def identity(x):
        if False:
            return 10
        return x

    @ray.remote
    def average(x):
        if False:
            for i in range(10):
                print('nop')
        return np.mean(x)

    @ray.remote
    def simple_large_intermediate():
        if False:
            while True:
                i = 10
        x = large_input.bind()
        y = identity.bind(x)
        return workflow.continuation(average.bind(y))
    start = time.time()
    outputs = workflow.run(simple_large_intermediate.bind())
    print(f'duration = {time.time() - start}')
    assert np.isclose(outputs, 8388607.5)
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))