import pytest
import ray
from ray import workflow
from filelock import FileLock

def test_workflow_manager_simple(workflow_start_regular):
    if False:
        i = 10
        return i + 15
    from ray.workflow.exceptions import WorkflowNotFoundError
    assert [] == workflow.list_all()
    with pytest.raises(WorkflowNotFoundError):
        workflow.get_status('X')

def test_workflow_manager(workflow_start_regular, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    tmp_file = str(tmp_path / 'lock')
    lock = FileLock(tmp_file)
    lock.acquire()
    flag_file = tmp_path / 'flag'
    flag_file.touch()

    @ray.remote
    def long_running(i):
        if False:
            for i in range(10):
                print('nop')
        lock = FileLock(tmp_file)
        with lock.acquire():
            pass
        if i % 2 == 0:
            if flag_file.exists():
                raise ValueError()
        return 100
    outputs = [workflow.run_async(long_running.bind(i), workflow_id=str(i)) for i in range(100)]
    all_tasks = workflow.list_all()
    assert len(all_tasks) == 100
    all_tasks_running = workflow.list_all(workflow.RUNNING)
    assert dict(all_tasks) == dict(all_tasks_running)
    assert workflow.get_status('0') == 'RUNNING'
    lock.release()
    for o in outputs:
        try:
            r = ray.get(o)
        except Exception:
            continue
        assert 100 == r
    all_tasks_running = workflow.list_all(workflow.WorkflowStatus.RUNNING)
    assert len(all_tasks_running) == 0
    failed_jobs = workflow.list_all('FAILED')
    assert len(failed_jobs) == 50
    finished_jobs = workflow.list_all('SUCCESSFUL')
    assert len(finished_jobs) == 50
    all_tasks_status = workflow.list_all({workflow.WorkflowStatus.SUCCESSFUL, workflow.WorkflowStatus.FAILED, workflow.WorkflowStatus.RUNNING})
    assert len(all_tasks_status) == 100
    assert failed_jobs == [(k, v) for (k, v) in all_tasks_status if v == workflow.WorkflowStatus.FAILED]
    assert finished_jobs == [(k, v) for (k, v) in all_tasks_status if v == workflow.WorkflowStatus.SUCCESSFUL]
    assert workflow.get_status('0') == 'FAILED'
    assert workflow.get_status('1') == 'SUCCESSFUL'
    lock.acquire()
    r = workflow.resume_async('0')
    assert workflow.get_status('0') == workflow.RUNNING
    flag_file.unlink()
    lock.release()
    assert 100 == ray.get(r)
    assert workflow.get_status('0') == workflow.SUCCESSFUL
    lock.acquire()
    workflow.resume_async('2')
    assert workflow.get_status('2') == workflow.RUNNING
    workflow.cancel('2')
    assert workflow.get_status('2') == workflow.CANCELED
    resumed = workflow.resume_all(include_failed=True)
    assert len(resumed) == 48
    lock.release()
    assert [ray.get(o) for (_, o) in resumed] == [100] * 48
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))