from ray.tests.conftest import *
import pytest
import ray
from ray import workflow
from ray.workflow.common import WorkflowRef

def test_dynamic_workflow_ref(workflow_start_regular_shared):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    def incr(x):
        if False:
            print('Hello World!')
        return x + 1
    assert workflow.run(incr.bind(0), workflow_id='test_dynamic_workflow_ref') == 1
    assert workflow.run(incr.bind(WorkflowRef('incr')), workflow_id='test_dynamic_workflow_ref') == 1
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))