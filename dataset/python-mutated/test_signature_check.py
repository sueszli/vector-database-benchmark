import pytest
from ray.tests.conftest import *
import ray
from ray import workflow

@ray.remote
def signature_check(a, b, c=1):
    if False:
        return 10
    pass

def test_signature_check(workflow_start_regular):
    if False:
        while True:
            i = 10
    with pytest.raises(TypeError):
        signature_check(1, 2)
    with pytest.raises(TypeError):
        workflow.run(signature_check.bind(1))
    with pytest.raises(TypeError):
        workflow.run(signature_check.bind(1, c=2))
    with pytest.raises(TypeError):
        workflow.run(signature_check.bind(1, 2, d=3))
    with pytest.raises(TypeError):
        workflow.run(signature_check.bind(1, 2, 3, 4))
    workflow.run(signature_check.bind(1, 2, 3))
    workflow.run(signature_check.bind(1, 2, c=3))
    workflow.run(signature_check.bind(1, b=2, c=3))
    workflow.run(signature_check.bind(a=1, b=2, c=3))
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))