import os
import sys
import threading
from time import sleep
import pytest
from ray._private.resource_spec import HEAD_NODE_RESOURCE_NAME
from ray._private.test_utils import wait_for_condition
from ray.tests.conftest_docker import *
scripts = '\nimport json\nimport os\n\nimport ray\n\nfrom ray import serve\n\n@serve.deployment\nclass GetPID:\n    def __call__(self, *args):\n        return {{"pid": os.getpid()}}\n\nserve.run(GetPID.options(num_replicas={num_replicas}).bind())\n'
check_script = '\nimport ray\nimport requests\n\n@ray.remote\ndef get_pid():\n    return requests.get("http://127.0.0.1:8000/").json()["pid"]\n\npids = {{\n    requests.get("http://127.0.0.1:8000/").json()["pid"]\n    for _ in range(20)\n}}\nprint(pids)\nassert len(pids) == {num_replicas}\n'
check_ray_nodes_script = '\nimport ray\n\nray.init(address="auto")\nprint(ray.nodes())\n'

@pytest.mark.skipif(sys.platform != 'linux', reason='Only works on linux.')
def test_ray_serve_basic(docker_cluster):
    if False:
        while True:
            i = 10
    (head, worker) = docker_cluster
    output = worker.exec_run(cmd=f"python -c '{scripts.format(num_replicas=1)}'")
    assert output.exit_code == 0, output.output
    assert b'Adding 1 replica to deployment ' in output.output
    output = worker.exec_run(cmd=f"python -c '{check_script.format(num_replicas=1)}'")
    assert output.exit_code == 0, output.output
    head.kill()
    output = worker.exec_run(cmd=f"python -c '{check_script.format(num_replicas=1)}'")
    assert output.exit_code == 0, output.output

    def reconfig():
        if False:
            for i in range(10):
                print('nop')
        worker.exec_run(cmd=f"python -c '{scripts.format(num_replicas=2)}'")
    t = threading.Thread(target=reconfig)
    t.start()
    sleep(1)
    head.restart()
    t.join()

    def check_for_head_node_come_back_up():
        if False:
            while True:
                i = 10
        _output = head.exec_run(cmd=f"python -c '{check_ray_nodes_script}'")
        return _output.exit_code == 0 and bytes(HEAD_NODE_RESOURCE_NAME, 'utf-8') in _output.output
    wait_for_condition(check_for_head_node_come_back_up)
    output = worker.exec_run(cmd=f"python -c '{check_script.format(num_replicas=2)}'")
    assert output.exit_code == 0, output.output
    check_controller_head_node_script = '\nimport ray\nimport requests\nfrom ray.serve.schema import ServeInstanceDetails\nfrom ray._private.resource_spec import HEAD_NODE_RESOURCE_NAME\nray.init(address="auto")\nhead_node_id = ray.get_runtime_context().get_node_id()\nserve_details = ServeInstanceDetails(\n    **requests.get("http://localhost:52365/api/serve/applications/").json())\nassert serve_details.controller_info.node_id == head_node_id\n'
    output = head.exec_run(cmd=f"python -c '{check_controller_head_node_script}'")
    assert output.exit_code == 0, output.output
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))