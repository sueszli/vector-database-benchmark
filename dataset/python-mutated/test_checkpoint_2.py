import ray
import time
import pytest
from ray.tests.conftest import *
import numpy as np
from ray import workflow
from ray.workflow.tests import utils
SIZE = 2 ** 15

@ray.remote
def checkpoint_dag(checkpoint):
    if False:
        i = 10
        return i + 15

    @ray.remote
    def large_input():
        if False:
            for i in range(10):
                print('nop')
        return np.arange(SIZE)

    @ray.remote
    def identity(x):
        if False:
            return 10
        if not utils.check_global_mark():
            import os
            os.kill(os.getpid(), 9)
        return x

    @ray.remote
    def average(x):
        if False:
            print('Hello World!')
        return np.mean(x)
    x = large_input.options(**workflow.options(checkpoint=checkpoint)).bind()
    y = identity.options(**workflow.options(checkpoint=checkpoint)).bind(x)
    return workflow.continuation(average.bind(y))

def test_checkpoint_dag_recovery_skip(workflow_start_regular_shared):
    if False:
        while True:
            i = 10
    utils.unset_global_mark()
    start = time.time()
    with pytest.raises(workflow.WorkflowExecutionError):
        workflow.run(checkpoint_dag.options(**workflow.options(checkpoint=False)).bind(False), workflow_id='checkpoint_skip_recovery')
    run_duration_skipped = time.time() - start
    utils.set_global_mark()
    start = time.time()
    recovered = workflow.resume('checkpoint_skip_recovery')
    recover_duration_skipped = time.time() - start
    assert np.isclose(recovered, np.arange(SIZE).mean())
    print(f'[skipped] run_duration = {run_duration_skipped}, recover_duration = {recover_duration_skipped}')

def test_checkpoint_dag_recovery_partial(workflow_start_regular_shared):
    if False:
        return 10
    utils.unset_global_mark()
    start = time.time()
    with pytest.raises(workflow.WorkflowExecutionError):
        workflow.run(checkpoint_dag.bind(False), workflow_id='checkpoint_partial_recovery')
    run_duration_partial = time.time() - start
    utils.set_global_mark()
    start = time.time()
    recovered = workflow.resume('checkpoint_partial_recovery')
    recover_duration_partial = time.time() - start
    assert np.isclose(recovered, np.arange(SIZE).mean())
    print(f'[partial] run_duration = {run_duration_partial}, recover_duration = {recover_duration_partial}')

def test_checkpoint_dag_recovery_whole(workflow_start_regular_shared):
    if False:
        print('Hello World!')
    utils.unset_global_mark()
    start = time.time()
    with pytest.raises(workflow.WorkflowExecutionError):
        workflow.run(checkpoint_dag.bind(True), workflow_id='checkpoint_whole_recovery')
    run_duration_whole = time.time() - start
    utils.set_global_mark()
    start = time.time()
    recovered = workflow.resume('checkpoint_whole_recovery')
    recover_duration_whole = time.time() - start
    assert np.isclose(recovered, np.arange(SIZE).mean())
    print(f'[whole] run_duration = {run_duration_whole}, recover_duration = {recover_duration_whole}')

@pytest.mark.skip(reason='Currently it is not clear how and if we need to checkside effects of skipping checkpointing, e.g., theviolation of exactly-once execution guarantee of workflow.')
def test_checkpoint_dag_validation(workflow_start_regular):
    if False:
        print('Hello World!')

    @ray.remote
    def identity(x):
        if False:
            return 10
        return x

    @ray.remote
    def average(x):
        if False:
            while True:
                i = 10
        return np.mean(x)

    @workflow.task
    def valid_checkpoint_dag_1():
        if False:
            return 10
        y = identity.options(checkpoint=False).task(42)
        return average.options(checkpoint=True).task(y)

    @workflow.task
    def invalid_checkpoint_dag_1():
        if False:
            while True:
                i = 10
        y = identity.options(checkpoint=True).task(42)
        return average.options(checkpoint=True).task(y)

    @workflow.task
    def invalid_checkpoint_dag_2():
        if False:
            return 10
        y = valid_checkpoint_dag_1.options(checkpoint=False).bind()
        return average.options(checkpoint=True).task(y)
    valid_checkpoint_dag_1.options(checkpoint=False).bind().run()
    with pytest.raises(workflow.WorkflowExecutionError):
        invalid_checkpoint_dag_1.options(checkpoint=False).bind().run()
    with pytest.raises(workflow.WorkflowExecutionError):
        invalid_checkpoint_dag_2.options(checkpoint=False).bind().run()
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))