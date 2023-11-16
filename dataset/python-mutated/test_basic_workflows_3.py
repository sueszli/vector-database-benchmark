import pytest
from filelock import FileLock
from pathlib import Path
import ray
from ray import workflow
from ray.tests.conftest import *

def test_wf_run(workflow_start_regular_shared, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    counter = tmp_path / 'counter'
    counter.write_text('0')

    @ray.remote
    def f():
        if False:
            return 10
        v = int(counter.read_text()) + 1
        counter.write_text(str(v))
    workflow.run(f.bind(), workflow_id='abc')
    assert counter.read_text() == '1'
    workflow.run(f.bind(), workflow_id='abc')
    assert counter.read_text() == '1'

def test_dedupe_indirect(workflow_start_regular_shared, tmp_path):
    if False:
        print('Hello World!')
    counter = Path(tmp_path) / 'counter.txt'
    lock = Path(tmp_path) / 'lock.txt'
    counter.write_text('0')

    @ray.remote
    def incr():
        if False:
            for i in range(10):
                print('nop')
        with FileLock(str(lock)):
            c = int(counter.read_text())
            c += 1
            counter.write_text(f'{c}')

    @ray.remote
    def identity(a):
        if False:
            for i in range(10):
                print('nop')
        return a

    @ray.remote
    def join(*a):
        if False:
            for i in range(10):
                print('nop')
        return counter.read_text()
    a = incr.bind()
    i1 = identity.bind(a)
    i2 = identity.bind(a)
    assert '1' == workflow.run(join.bind(i1, i2))
    assert '2' == workflow.run(join.bind(i1, i2))
    assert '3' == workflow.run(join.bind(a, a, a, a))
    assert '4' == workflow.run(join.bind(a, a, a, a))

def test_run_off_main_thread(workflow_start_regular_shared):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    def fake_data(num: int):
        if False:
            i = 10
            return i + 15
        return list(range(num))
    succ = False

    def run():
        if False:
            print('Hello World!')
        global succ
        assert workflow.run(fake_data.bind(10), workflow_id='run') == list(range(10))
    import threading
    t = threading.Thread(target=run)
    t.start()
    t.join()
    assert workflow.get_status('run') == workflow.SUCCESSFUL

def test_task_id_generation(workflow_start_regular_shared, request):
    if False:
        return 10

    @ray.remote
    def simple(x):
        if False:
            return 10
        return x + 1
    x = simple.options(**workflow.options(task_id='simple')).bind(-1)
    n = 20
    for i in range(1, n):
        x = simple.options(**workflow.options(task_id='simple')).bind(x)
    workflow_id = 'test_task_id_generation'
    ret = workflow.run_async(x, workflow_id=workflow_id)
    outputs = [workflow.get_output_async(workflow_id, task_id='simple')]
    for i in range(1, n):
        outputs.append(workflow.get_output_async(workflow_id, task_id=f'simple_{i}'))
    assert ray.get(ret) == n - 1
    assert ray.get(outputs) == list(range(n))
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))