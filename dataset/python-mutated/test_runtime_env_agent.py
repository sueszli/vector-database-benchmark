import sys
import pytest
import logging
import os
import time
from typing import List, Tuple
import ray
from ray._private.runtime_env.agent.runtime_env_agent import UriType, ReferenceTable
from ray._private import ray_constants
from ray._private.test_utils import get_error_message, init_error_pubsub, wait_for_condition
from ray.runtime_env import RuntimeEnv
import psutil
logger = logging.getLogger(__name__)

def test_reference_table():
    if False:
        while True:
            i = 10
    expected_unused_uris = []
    expected_unused_runtime_env = str()

    def uris_parser(runtime_env) -> Tuple[str, UriType]:
        if False:
            while True:
                i = 10
        result = list()
        result.append((runtime_env.working_dir(), 'working_dir'))
        py_module_uris = runtime_env.py_modules()
        for uri in py_module_uris:
            result.append((uri, 'py_modules'))
        return result

    def unused_uris_processor(unused_uris: List[Tuple[str, UriType]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        nonlocal expected_unused_uris
        assert expected_unused_uris
        for unused in unused_uris:
            assert unused in expected_unused_uris
            expected_unused_uris.remove(unused)
        assert not expected_unused_uris

    def unused_runtime_env_processor(unused_runtime_env: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        nonlocal expected_unused_runtime_env
        assert expected_unused_runtime_env
        assert expected_unused_runtime_env == unused_runtime_env
        expected_unused_runtime_env = None
    reference_table = ReferenceTable(uris_parser, unused_uris_processor, unused_runtime_env_processor)
    runtime_env_1 = RuntimeEnv(working_dir='s3://working_dir_1.zip', py_modules=['s3://py_module_A.zip', 's3://py_module_B.zip'])
    runtime_env_2 = RuntimeEnv(working_dir='s3://working_dir_2.zip', py_modules=['s3://py_module_A.zip', 's3://py_module_C.zip'])
    reference_table.increase_reference(runtime_env_1, runtime_env_1.serialize(), 'raylet')
    reference_table.increase_reference(runtime_env_2, runtime_env_2.serialize(), 'raylet')
    reference_table.increase_reference(runtime_env_1, runtime_env_1.serialize(), 'client_server')
    expected_unused_uris.append(('s3://working_dir_1.zip', 'working_dir'))
    expected_unused_uris.append(('s3://py_module_B.zip', 'py_modules'))
    expected_unused_runtime_env = runtime_env_1.serialize()
    reference_table.decrease_reference(runtime_env_1, runtime_env_1.serialize(), 'raylet')
    assert not expected_unused_uris
    assert not expected_unused_runtime_env
    expected_unused_uris.append(('s3://working_dir_2.zip', 'working_dir'))
    expected_unused_uris.append(('s3://py_module_A.zip', 'py_modules'))
    expected_unused_uris.append(('s3://py_module_C.zip', 'py_modules'))
    expected_unused_runtime_env = runtime_env_2.serialize()
    reference_table.decrease_reference(runtime_env_2, runtime_env_2.serialize(), 'raylet')
    assert not expected_unused_uris
    assert not expected_unused_runtime_env

def search_agent(processes):
    if False:
        i = 10
        return i + 15
    for p in processes:
        try:
            for c in p.cmdline():
                if os.path.join('runtime_env', 'agent', 'main.py') in c:
                    return p
        except Exception:
            pass

def check_agent_register(raylet_proc, agent_pid):
    if False:
        return 10
    for x in range(5):
        logger.info('Check agent is alive.')
        agent_proc = search_agent(raylet_proc.children())
        assert agent_proc.pid == agent_pid
        time.sleep(1)

@pytest.mark.skipif(sys.platform == 'win32', reason='no fate sharing for windows')
def test_raylet_and_agent_share_fate(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    'Test raylet and agent share fate.'
    ray.init()
    p = init_error_pubsub()
    node = ray._private.worker._global_node
    all_processes = node.all_processes
    raylet_proc_info = all_processes[ray_constants.PROCESS_TYPE_RAYLET][0]
    raylet_proc = psutil.Process(raylet_proc_info.process.pid)
    wait_for_condition(lambda : search_agent(raylet_proc.children()))
    agent_proc = search_agent(raylet_proc.children())
    agent_pid = agent_proc.pid
    check_agent_register(raylet_proc, agent_pid)
    raylet_proc.terminate()
    raylet_proc.wait()
    agent_proc.wait(15)
    errors = get_error_message(p, 1, ray_constants.RAYLET_DIED_ERROR)
    assert len(errors) == 0, errors
    ray.shutdown()
    ray.init()
    all_processes = ray._private.worker._global_node.all_processes
    raylet_proc_info = all_processes[ray_constants.PROCESS_TYPE_RAYLET][0]
    raylet_proc = psutil.Process(raylet_proc_info.process.pid)
    wait_for_condition(lambda : search_agent(raylet_proc.children()))
    agent_proc = search_agent(raylet_proc.children())
    agent_pid = agent_proc.pid
    check_agent_register(raylet_proc, agent_pid)
    agent_proc.kill()
    agent_proc.wait()
    raylet_proc.wait(15)

@pytest.mark.skipif(sys.platform == 'win32', reason='no fate sharing for windows')
def test_agent_report_unexpected_raylet_death(shutdown_only):
    if False:
        print('Hello World!')
    'Test agent reports Raylet death if it is not SIGTERM.'
    ray.init()
    p = init_error_pubsub()
    node = ray._private.worker._global_node
    all_processes = node.all_processes
    raylet_proc_info = all_processes[ray_constants.PROCESS_TYPE_RAYLET][0]
    raylet_proc = psutil.Process(raylet_proc_info.process.pid)
    wait_for_condition(lambda : search_agent(raylet_proc.children()))
    agent_proc = search_agent(raylet_proc.children())
    agent_pid = agent_proc.pid
    check_agent_register(raylet_proc, agent_pid)
    raylet_proc.kill()
    raylet_proc.wait()
    agent_proc.wait(15)
    errors = get_error_message(p, 1, ray_constants.RAYLET_DIED_ERROR)
    assert len(errors) == 1, errors
    err = errors[0]
    assert err['type'] == ray_constants.RAYLET_DIED_ERROR
    assert 'Termination is unexpected.' in err['error_message'], err['error_message']
    assert 'Raylet logs:' in err['error_message'], err['error_message']
    assert os.path.getsize(os.path.join(node.get_session_dir_path(), 'logs', 'raylet.out')) < 1 * 1024 ** 2

@pytest.mark.skipif(sys.platform == 'win32', reason='no fate sharing for windows')
def test_agent_report_unexpected_raylet_death_large_file(shutdown_only):
    if False:
        print('Hello World!')
    'Test agent reports Raylet death if it is not SIGTERM.'
    ray.init()
    p = init_error_pubsub()
    node = ray._private.worker._global_node
    all_processes = node.all_processes
    raylet_proc_info = all_processes[ray_constants.PROCESS_TYPE_RAYLET][0]
    raylet_proc = psutil.Process(raylet_proc_info.process.pid)
    wait_for_condition(lambda : search_agent(raylet_proc.children()))
    agent_proc = search_agent(raylet_proc.children())
    agent_pid = agent_proc.pid
    check_agent_register(raylet_proc, agent_pid)
    with open(os.path.join(node.get_session_dir_path(), 'logs', 'raylet.out'), 'a') as f:
        f.write('test data\n' * 1024 ** 2)
    raylet_proc.kill()
    raylet_proc.wait()
    agent_proc.wait(15)
    errors = get_error_message(p, 1, ray_constants.RAYLET_DIED_ERROR)
    assert len(errors) == 1, errors
    err = errors[0]
    assert err['type'] == ray_constants.RAYLET_DIED_ERROR
    assert 'Termination is unexpected.' in err['error_message'], err['error_message']
    assert 'Raylet logs:' in err['error_message'], err['error_message']
if __name__ == '__main__':
    sys.exit(pytest.main(['-v', __file__]))