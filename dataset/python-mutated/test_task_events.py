from collections import defaultdict
from typing import Dict
import os
import pytest
import sys
import threading
import time
from ray._private.state_api_test_utils import verify_failed_task
from ray.exceptions import RuntimeEnvSetupError
from ray.runtime_env import RuntimeEnv
import ray
from ray.util.state.common import ListApiOptions, StateResource
from ray._private.test_utils import raw_metrics, run_string_as_driver, run_string_as_driver_nonblocking, wait_for_condition
from ray.util.state import StateApiClient, list_tasks
from ray._private.worker import RayContext
_SYSTEM_CONFIG = {'task_events_report_interval_ms': 100, 'metrics_report_interval_ms': 200, 'enable_timeline': False, 'gcs_mark_task_failed_on_job_done_delay_ms': 1000}

def aggregate_task_event_metric(info: RayContext) -> Dict:
    if False:
        return 10
    '\n    Aggregate metrics of task events into:\n        {\n            "REPORTED": ray_gcs_task_manager_task_events_reported,\n            "STORED": ray_gcs_task_manager_task_events_stored,\n            "DROPPED_PROFILE_EVENT":\n                ray_gcs_task_manager_task_events_dropped PROFILE_EVENT,\n            "DROPPED_STATUS_EVENT":\n                ray_gcs_task_manager_task_events_dropped STATUS_EVENT,\n        }\n    '
    res = raw_metrics(info)
    task_events_info = defaultdict(int)
    if 'ray_gcs_task_manager_task_events_dropped' in res:
        for sample in res['ray_gcs_task_manager_task_events_dropped']:
            if 'Type' in sample.labels and sample.labels['Type'] != '':
                task_events_info['DROPPED_' + sample.labels['Type']] += sample.value
    if 'ray_gcs_task_manager_task_events_stored' in res:
        for sample in res['ray_gcs_task_manager_task_events_stored']:
            task_events_info['STORED'] += sample.value
    if 'ray_gcs_task_manager_task_events_reported' in res:
        for sample in res['ray_gcs_task_manager_task_events_reported']:
            task_events_info['REPORTED'] += sample.value
    return task_events_info

def test_status_task_events_metrics(shutdown_only):
    if False:
        return 10
    info = ray.init(num_cpus=1, _system_config=_SYSTEM_CONFIG)

    @ray.remote
    def f():
        if False:
            i = 10
            return i + 15
        pass
    for _ in range(10):
        ray.get(f.remote())

    def verify():
        if False:
            for i in range(10):
                print('nop')
        metric = aggregate_task_event_metric(info)
        assert metric['REPORTED'] >= 10, 'At least 10 tasks events should be reported. Could be more than 10 with multiple flush.'
        assert metric['STORED'] == 11, "10 task + 1 driver's events should be stored."
        return True
    wait_for_condition(verify, timeout=20, retry_interval_ms=100)

def test_failed_task_error(shutdown_only):
    if False:
        while True:
            i = 10
    ray.init(_system_config=_SYSTEM_CONFIG)
    error_msg_str = 'fail is expected to fail'

    @ray.remote
    def fail(x=None):
        if False:
            while True:
                i = 10
        if x is not None:
            time.sleep(x)
        raise ValueError(error_msg_str)
    with pytest.raises(ray.exceptions.RayTaskError):
        ray.get(fail.options(name='fail').remote())
    wait_for_condition(verify_failed_task, name='fail', error_type='TASK_EXECUTION_EXCEPTION', error_message=error_msg_str)

    @ray.remote
    def not_running():
        if False:
            return 10
        raise ValueError('should not be run')
    with pytest.raises(ray.exceptions.TaskCancelledError):
        t = not_running.options(name='cancel-before-running').remote()
        ray.cancel(t)
        ray.get(t)
    wait_for_condition(verify_failed_task, name='cancel-before-running', error_type='TASK_CANCELLED', error_message='')

    @ray.remote(max_retries=0)
    def die():
        if False:
            print('Hello World!')
        exit(27)
    with pytest.raises(ray.exceptions.WorkerCrashedError):
        ray.get(die.options(name='die-worker').remote())
    wait_for_condition(verify_failed_task, name='die-worker', error_type='WORKER_DIED', error_message='Worker exits with an exit code 27')

    @ray.remote
    class Actor:

        def f(self):
            if False:
                return 10
            time.sleep(999)

        def ready(self):
            if False:
                i = 10
                return i + 15
            pass
    a = Actor.remote()
    ray.get(a.ready.remote())
    with pytest.raises(ray.exceptions.RayActorError):
        ray.kill(a)
        ray.get(a.f.options(name='actor-killed').remote())
    wait_for_condition(verify_failed_task, name='actor-killed', error_type='ACTOR_DIED', error_message='The actor is dead because it was killed by `ray.kill`')

def test_failed_task_failed_due_to_node_failure(ray_start_cluster):
    if False:
        return 10
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=1)
    ray.init(address=cluster.address)
    node = cluster.add_node(num_cpus=2)
    driver_script = '\nimport ray\nray.init("auto")\n\n@ray.remote(num_cpus=2, max_retries=0)\ndef sleep():\n    import time\n    time.sleep(999)\n\nx = sleep.options(name="node-killed").remote()\nray.get(x)\n    '
    run_string_as_driver_nonblocking(driver_script)

    def driver_running():
        if False:
            while True:
                i = 10
        t = list_tasks(filters=[('name', '=', 'node-killed')])
        return len(t) > 0
    wait_for_condition(driver_running)
    cluster.remove_node(node)
    wait_for_condition(verify_failed_task, name='node-killed', error_type='NODE_DIED', error_message='Task failed due to the node dying')

def test_failed_task_unschedulable(shutdown_only):
    if False:
        return 10
    ray.init(num_cpus=1, _system_config=_SYSTEM_CONFIG)
    node_id = ray.get_runtime_context().get_node_id()
    policy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)

    @ray.remote
    def task():
        if False:
            print('Hello World!')
        pass
    task.options(scheduling_strategy=policy, name='task-unschedulable', num_cpus=2).remote()
    wait_for_condition(verify_failed_task, name='task-unschedulable', error_type='TASK_UNSCHEDULABLE_ERROR', error_message="The node specified via NodeAffinitySchedulingStrategy doesn't exist any more or is infeasible")

def test_failed_task_runtime_env_setup(shutdown_only):
    if False:
        while True:
            i = 10

    @ray.remote
    def f():
        if False:
            for i in range(10):
                print('nop')
        pass
    bad_env = RuntimeEnv(conda={'dependencies': ['_this_does_not_exist']})
    with pytest.raises(RuntimeEnvSetupError):
        ray.get(f.options(runtime_env=bad_env, name='task-runtime-env-failed').remote())
    wait_for_condition(verify_failed_task, name='task-runtime-env-failed', error_type='RUNTIME_ENV_SETUP_FAILED', error_message='ResolvePackageNotFound')

def test_parent_task_id_threaded_task(shutdown_only):
    if False:
        print('Hello World!')
    ray.init(_system_config=_SYSTEM_CONFIG)

    @ray.remote
    def main_task():
        if False:
            i = 10
            return i + 15

        def thd_task():
            if False:
                i = 10
                return i + 15

            @ray.remote
            def thd_task():
                if False:
                    return 10
                pass
            ray.get(thd_task.remote())
        thd = threading.Thread(target=thd_task)
        thd.start()
        thd.join()
    ray.get(main_task.remote())

    def verify():
        if False:
            return 10
        tasks = list_tasks()
        assert len(tasks) == 2
        expect_parent_task_id = None
        actual_parent_task_id = None
        for task in tasks:
            if task['name'] == 'main_task':
                expect_parent_task_id = task['task_id']
            elif task['name'] == 'thd_task':
                actual_parent_task_id = task['parent_task_id']
        assert actual_parent_task_id is not None
        assert expect_parent_task_id == actual_parent_task_id
        return True
    wait_for_condition(verify)

def test_parent_task_id_non_concurrent_actor(shutdown_only):
    if False:
        return 10
    ray.init(_system_config=_SYSTEM_CONFIG)

    def run_task_in_thread():
        if False:
            return 10

        def thd_task():
            if False:
                return 10

            @ray.remote
            def thd_task():
                if False:
                    print('Hello World!')
                pass
            ray.get(thd_task.remote())
        thd = threading.Thread(target=thd_task)
        thd.start()
        thd.join()

    @ray.remote
    class Actor:

        def main_task(self):
            if False:
                return 10
            run_task_in_thread()
    a = Actor.remote()
    ray.get(a.main_task.remote())

    def verify():
        if False:
            for i in range(10):
                print('nop')
        tasks = list_tasks()
        expect_parent_task_id = None
        actual_parent_task_id = None
        for task in tasks:
            if 'main_task' in task['name']:
                expect_parent_task_id = task['task_id']
            elif 'thd_task' in task['name']:
                actual_parent_task_id = task['parent_task_id']
        print(tasks)
        assert actual_parent_task_id is not None
        assert expect_parent_task_id == actual_parent_task_id
        return True
    wait_for_condition(verify)

@pytest.mark.parametrize('actor_concurrency', [3, 10])
def test_parent_task_id_concurrent_actor(shutdown_only, actor_concurrency):
    if False:
        return 10
    ray.init(_system_config=_SYSTEM_CONFIG)

    def run_task_in_thread(name, i):
        if False:
            print('Hello World!')

        def thd_task():
            if False:
                return 10

            @ray.remote
            def thd_task():
                if False:
                    i = 10
                    return i + 15
                pass
            ray.get(thd_task.options(name=f'{name}_{i}').remote())
        thd = threading.Thread(target=thd_task)
        thd.start()
        thd.join()

    @ray.remote
    class AsyncActor:

        async def main_task(self, i):
            run_task_in_thread('async_thd_task', i)

    @ray.remote
    class ThreadedActor:

        def main_task(self, i):
            if False:
                i = 10
                return i + 15
            run_task_in_thread('threaded_thd_task', i)

    def verify(actor_method_name, actor_class_name):
        if False:
            while True:
                i = 10
        tasks = list_tasks()
        print(tasks)
        expect_parent_task_id = None
        actual_parent_task_id = None
        for task in tasks:
            if f'{actor_class_name}.__init__' in task['name']:
                expect_parent_task_id = task['task_id']
        assert expect_parent_task_id is not None
        for task in tasks:
            if f'{actor_method_name}' in task['name']:
                actual_parent_task_id = task['parent_task_id']
                assert expect_parent_task_id == actual_parent_task_id, task
        return True
    async_actor = AsyncActor.options(max_concurrency=actor_concurrency).remote()
    ray.get([async_actor.main_task.remote(i) for i in range(20)])
    wait_for_condition(verify, actor_class_name='AsyncActor', actor_method_name='async_thd_task')
    thd_actor = ThreadedActor.options(max_concurrency=actor_concurrency).remote()
    ray.get([thd_actor.main_task.remote(i) for i in range(20)])
    wait_for_condition(verify, actor_class_name='ThreadedActor', actor_method_name='threaded_thd_task')

def test_parent_task_id_tune_e2e(shutdown_only):
    if False:
        print('Hello World!')
    ray.init(_system_config=_SYSTEM_CONFIG)
    job_id = ray.get_runtime_context().get_job_id()
    script = '\nimport numpy as np\nimport ray\nimport ray.train\nfrom ray import tune\nimport time\n\nray.init("auto")\n\n@ray.remote\ndef train_step_1():\n    time.sleep(0.5)\n    return 1\n\ndef train_function(config):\n    for i in range(5):\n        loss = config["mean"] * np.random.randn() + ray.get(\n            train_step_1.remote())\n        ray.train.report(dict(loss=loss, nodes=ray.nodes()))\n\n\ndef tune_function():\n    analysis = tune.run(\n        train_function,\n        metric="loss",\n        mode="min",\n        config={\n            "mean": tune.grid_search([1, 2, 3, 4, 5]),\n        },\n        resources_per_trial=tune.PlacementGroupFactory([{\n            \'CPU\': 1.0\n        }] + [{\n            \'CPU\': 1.0\n        }] * 3),\n    )\n    return analysis.best_config\n\n\ntune_function()\n    '
    run_string_as_driver(script)
    client = StateApiClient()

    def list_tasks():
        if False:
            while True:
                i = 10
        return client.list(StateResource.TASKS, options=ListApiOptions(exclude_driver=False, filters=[('job_id', '!=', job_id)], limit=1000), raise_on_missing_output=True)

    def verify():
        if False:
            i = 10
            return i + 15
        tasks = list_tasks()
        task_id_map = {task['task_id']: task for task in tasks}
        for task in tasks:
            if task['type'] == 'DRIVER_TASK':
                continue
            assert task_id_map.get(task['parent_task_id'], None) is not None, task
        return True
    wait_for_condition(verify)

def test_is_debugger_paused(shutdown_only):
    if False:
        i = 10
        return i + 15
    ray.init(num_cpus=1, _system_config=_SYSTEM_CONFIG)

    @ray.remote(max_retries=0)
    def f():
        if False:
            print('Hello World!')
        import time
        with ray._private.worker.global_worker.task_paused_by_debugger():
            time.sleep(5)

    def verify(num_paused):
        if False:
            print('Hello World!')
        tasks = list_tasks(filters=[('is_debugger_paused', '=', 'True')])
        return len(tasks) == num_paused
    f_task = f.remote()
    wait_for_condition(verify, timeout=20, retry_interval_ms=100, num_paused=1)
    wait_for_condition(verify, timeout=20, retry_interval_ms=100, num_paused=0)

@pytest.mark.parametrize('actor_concurrency', [1, 3])
def test_is_debugger_paused_actor(shutdown_only, actor_concurrency):
    if False:
        for i in range(10):
            print('nop')
    ray.init(_system_config=_SYSTEM_CONFIG)

    @ray.remote
    class TestActor:

        def main_task(self, i):
            if False:
                return 10
            if i == 0:
                import time
                with ray._private.worker.global_worker.task_paused_by_debugger():
                    time.sleep(5)

    def verify(expected_task_name):
        if False:
            print('Hello World!')
        tasks = list_tasks(filters=[('is_debugger_paused', '=', 'True')])
        return len(tasks) == 1 and f'{expected_task_name}_0' in tasks[0]['name']
    test_actor = TestActor.options(max_concurrency=actor_concurrency).remote()
    refs = [test_actor.main_task.options(name=f'TestActor.main_task_{i}').remote(i) for i in range(20)]
    wait_for_condition(verify, expected_task_name='TestActor.main_task')

@pytest.mark.parametrize('actor_concurrency', [1, 3])
def test_is_debugger_paused_threaded_actor(shutdown_only, actor_concurrency):
    if False:
        for i in range(10):
            print('nop')
    ray.init(_system_config=_SYSTEM_CONFIG)

    @ray.remote
    class ThreadedActor:

        def main_task(self, i):
            if False:
                return 10

            def thd_task():
                if False:
                    while True:
                        i = 10

                @ray.remote
                def thd_task():
                    if False:
                        for i in range(10):
                            print('nop')
                    if i == 0:
                        import time
                        with ray._private.worker.global_worker.task_paused_by_debugger():
                            time.sleep(5)
                ray.get(thd_task.options(name=f'ThreadedActor.main_task_{i}').remote())
            thd = threading.Thread(target=thd_task)
            thd.start()
            thd.join()

    def verify(expected_task_name):
        if False:
            for i in range(10):
                print('nop')
        tasks = list_tasks(filters=[('is_debugger_paused', '=', 'True')])
        return len(tasks) == 1 and f'{expected_task_name}_0' in tasks[0]['name']
    threaded_actor = ThreadedActor.options(max_concurrency=actor_concurrency).remote()
    refs = [threaded_actor.main_task.options(name=f'ThreadedActor.main_task_{i}').remote(i) for i in range(20)]
    wait_for_condition(verify, expected_task_name='ThreadedActor.main_task')

@pytest.mark.parametrize('actor_concurrency', [1, 3])
def test_is_debugger_paused_async_actor(shutdown_only, actor_concurrency):
    if False:
        for i in range(10):
            print('nop')
    ray.init(_system_config=_SYSTEM_CONFIG)

    @ray.remote
    class AsyncActor:

        async def main_task(self, i):
            if i == 0:
                import time
                print()
                with ray._private.worker.global_worker.task_paused_by_debugger():
                    time.sleep(5)

    def verify(expected_task_name):
        if False:
            print('Hello World!')
        tasks = list_tasks(filters=[('is_debugger_paused', '=', 'True')])
        print(tasks)
        return len(tasks) == 1 and f'{expected_task_name}_0' in tasks[0]['name']
    async_actor = AsyncActor.options(max_concurrency=actor_concurrency).remote()
    refs = [async_actor.main_task.options(name=f'AsyncActor.main_task_{i}').remote(i) for i in range(20)]
    wait_for_condition(verify, expected_task_name='AsyncActor.main_task')
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))