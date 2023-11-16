"""Basic tests isolated from other tests for shared fixtures."""
import os
import pytest
from ray._private.test_utils import run_string_as_driver
import ray
from ray import workflow
from ray.tests.conftest import *

def test_workflow_error_message(shutdown_only):
    if False:
        print('Hello World!')
    storage_url = 'c:\\ray'
    expected_error_msg = f"Cannot parse URI: '{storage_url}'"
    if os.name == 'nt':
        expected_error_msg += ' Try using file://{} or file:///{} for Windows file paths.'.format(storage_url, storage_url)
    ray.shutdown()
    with pytest.raises(ValueError) as e:
        ray.init(storage=storage_url)
    assert str(e.value) == expected_error_msg

def test_options_update(shutdown_only):
    if False:
        i = 10
        return i + 15
    from ray.workflow.common import WORKFLOW_OPTIONS

    @workflow.options(task_id='old_name', metadata={'k': 'v'})
    @ray.remote(num_cpus=2, max_retries=1)
    def f():
        if False:
            print('Hello World!')
        return
    new_f = f.options(num_returns=2, **workflow.options(task_id='new_name', metadata={'extra_k2': 'extra_v2'}))
    options = new_f.bind().get_options()
    assert options == {'num_cpus': 2, 'num_returns': 2, 'max_retries': 1, '_metadata': {WORKFLOW_OPTIONS: {'task_id': 'new_name', 'metadata': {'extra_k2': 'extra_v2'}}}}

def test_no_init_run(shutdown_only):
    if False:
        i = 10
        return i + 15

    @ray.remote
    def f():
        if False:
            i = 10
            return i + 15
        pass
    workflow.run(f.bind())

def test_no_init_api(shutdown_only):
    if False:
        return 10
    workflow.list_all()

def test_object_valid(workflow_start_regular):
    if False:
        for i in range(10):
            print('nop')
    import uuid
    workflow_id = str(uuid.uuid4())
    script = f'\nimport ray\nfrom ray import workflow\nfrom typing import List\n\nray.init(address="{workflow_start_regular}")\n\n@ray.remote\ndef echo(data, sleep_s=0, others=None):\n    from time import sleep\n    sleep(sleep_s)\n    print(data)\n\na = {{"abc": "def"}}\ne1 = echo.bind(a, 5)\ne2 = echo.bind(a, 0, e1)\nworkflow.run_async(e2, workflow_id="{workflow_id}")\n'
    run_string_as_driver(script)
    print(ray.get(workflow.get_output_async(workflow_id=workflow_id)))
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))