import os
import ray
import time
import pytest
from ray._private.test_utils import run_string_as_driver_nonblocking, run_string_as_driver
from ray.tests.conftest import *
from ray import workflow
from unittest.mock import patch
driver_script = '\nimport time\nimport ray\nfrom ray import workflow\n\n\n@ray.remote\ndef foo(x):\n    time.sleep(1)\n    if x < 20:\n        return workflow.continuation(foo.bind(x + 1))\n    else:\n        return 20\n\n\nif __name__ == "__main__":\n    ray.init()\n    output = workflow.run_async(foo.bind(0), workflow_id="driver_terminated")\n    time.sleep({})\n'

def test_workflow_lifetime_1(workflow_start_cluster):
    if False:
        return 10
    (address, storage_uri) = workflow_start_cluster
    with patch.dict(os.environ, {'RAY_ADDRESS': address}):
        ray.init()
        run_string_as_driver(driver_script.format(5))
        assert workflow.get_output('driver_terminated') == 20

def test_workflow_lifetime_2(workflow_start_cluster):
    if False:
        for i in range(10):
            print('nop')
    (address, storage_uri) = workflow_start_cluster
    with patch.dict(os.environ, {'RAY_ADDRESS': address}):
        ray.init()
        proc = run_string_as_driver_nonblocking(driver_script.format(100))
        time.sleep(10)
        proc.kill()
        time.sleep(1)
        assert workflow.get_output('driver_terminated') == 20
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))