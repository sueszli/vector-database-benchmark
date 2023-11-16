from ray.tests.conftest import *
import ray
from ray import workflow
import pytest

@pytest.mark.skip(reason='Variable mutable is not supported right now.')
def test_variable_mutable(workflow_start_regular):
    if False:
        while True:
            i = 10

    @ray.remote
    def identity(x):
        if False:
            return 10
        return x

    @ray.remote
    def projection(x, _):
        if False:
            print('Hello World!')
        return x
    x = []
    a = identity.bind(x)
    x.append(1)
    b = identity.bind(x)
    assert workflow.run(projection.bind(a, b)) == []
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))