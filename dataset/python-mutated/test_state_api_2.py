import asyncio
import json
import os
import sys
from pathlib import Path
import tempfile
from collections import defaultdict
from ray._private.test_utils import check_call_subprocess
import ray
import requests
import pytest
from ray._private.profiling import chrome_tracing_dump
from ray.util.state import get_actor, list_tasks, list_actors, list_workers, list_nodes
from ray._private.test_utils import wait_for_condition

def test_timeline(shutdown_only):
    if False:
        print('Hello World!')
    ray.init(num_cpus=8)
    job_id = ray.get_runtime_context().get_job_id()
    TASK_SLEEP_TIME_S = 1

    @ray.remote
    def f():
        if False:
            while True:
                i = 10
        import time
        time.sleep(TASK_SLEEP_TIME_S)

    @ray.remote
    class Actor:

        def ready(self):
            if False:
                while True:
                    i = 10
            pass

    @ray.remote
    class AsyncActor:

        async def f(self):
            await asyncio.sleep(5)

        async def g(self):
            await asyncio.sleep(5)

    @ray.remote
    class ThreadedActor:

        def f(self):
            if False:
                for i in range(10):
                    print('nop')
            import time
            time.sleep(5)

        def g(self):
            if False:
                print('Hello World!')
            import time
            time.sleep(5)
    [f.remote() for _ in range(4)]
    a = Actor.remote()
    b = AsyncActor.remote()
    c = ThreadedActor.options(max_concurrency=15).remote()
    [a.ready.remote() for _ in range(4)]
    ray.get(b.f.remote())
    [b.f.remote() for _ in range(4)]
    [b.g.remote() for _ in range(4)]
    [c.f.remote() for _ in range(4)]
    [c.g.remote() for _ in range(4)]
    result = json.loads(chrome_tracing_dump(list_tasks(detail=True)))
    actor_to_events = defaultdict(list)
    task_to_events = defaultdict(list)
    index_to_workers = {}
    index_to_nodes = {}
    for item in result:
        if item['ph'] == 'M':
            name = item['name']
            if name == 'thread_name':
                index_to_workers[item['tid']] = item['args']['name']
            elif name == 'process_name':
                index_to_nodes[item['pid']] = item['args']['name']
            else:
                raise ValueError(f'Unexecpted name from metadata event {name}')
        elif item['ph'] == 'X':
            actor_id = item['args']['actor_id']
            assert 'actor_id' in item['args']
            assert 'attempt_number' in item['args']
            assert 'func_or_class_name' in item['args']
            assert 'job_id' in item['args']
            assert 'task_id' in item['args']
            if actor_id:
                actor_to_events[actor_id].append(item)
            else:
                task_to_events[item['args']['task_id']].append(item)
        else:
            raise ValueError(f"Unexpected event type {item['ph']}")
    actors = {actor['actor_id']: actor for actor in list_actors(detail=True)}
    tasks = {task['task_id']: task for task in list_tasks(detail=True)}
    workers = {worker['worker_id']: worker for worker in list_workers(detail=True)}
    nodes = {node['node_ip']: node for node in list_nodes(detail=True)}
    for (actor_id, events) in actor_to_events.items():
        for event in events:
            assert event['args']['actor_id'] == actor_id
            assert event['args']['job_id'] == job_id
            task_id = event['args']['task_id']
            assert event['args']['func_or_class_name'] == tasks[task_id]['func_or_class_name']
        worker_id_from_event = index_to_workers[event['tid']].split(':')[1]
        node_id_from_event = index_to_nodes[event['pid']].split(' ')[1]
        assert actors[actor_id]['pid'] == workers[worker_id_from_event]['pid']
        assert actors[actor_id]['node_id'] == nodes[node_id_from_event]['node_id']
    for (task_id, events) in task_to_events.items():
        for event in events:
            assert event['args']['job_id'] == job_id
            task_id = event['args']['task_id']
            assert event['args']['func_or_class_name'] == tasks[task_id]['func_or_class_name']
            if event['cat'] == 'task:execute':
                assert TASK_SLEEP_TIME_S * 1000000.0 * 0.9 < event['dur'] < TASK_SLEEP_TIME_S * 1000000.0 * 1.1
        worker_id_from_event = index_to_workers[event['tid']].split(':')[1]
        node_id_from_event = index_to_nodes[event['pid']].split(' ')[1]
        assert tasks[task_id]['worker_id'] == worker_id_from_event
        assert tasks[task_id]['node_id'] == nodes[node_id_from_event]['node_id']
    metadata_events = list(filter(lambda e: e['ph'] == 'M', result))
    assert len(metadata_events) == len(index_to_workers) + len(index_to_nodes)

def test_timeline_request(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    context = ray.init()
    dashboard_url = f"http://{context['webui_url']}"

    @ray.remote
    def f():
        if False:
            print('Hello World!')
        pass
    ray.get([f.remote() for _ in range(5)])

    def verify():
        if False:
            for i in range(10):
                print('nop')
        resp = requests.get(f'{dashboard_url}/api/v0/tasks/timeline')
        resp.raise_for_status()
        assert resp.json(), 'No result has returned'
        return True
    wait_for_condition(verify, timeout=10)

def test_actor_repr_name(shutdown_only):
    if False:
        for i in range(10):
            print('nop')

    def _verify_repr_name(id, name):
        if False:
            while True:
                i = 10
        actor = get_actor(id=id)
        assert actor is not None
        assert actor['repr_name'] == name
        return True

    @ray.remote
    class ReprActor:

        def __init__(self, x) -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.x = x

        def __repr__(self) -> str:
            if False:
                i = 10
                return i + 15
            return self.x

        def ready(self):
            if False:
                print('Hello World!')
            pass
    a = ReprActor.remote(x='repr-name-a')
    b = ReprActor.remote(x='repr-name-b')
    wait_for_condition(_verify_repr_name, id=a._actor_id.hex(), name='repr-name-a')
    wait_for_condition(_verify_repr_name, id=b._actor_id.hex(), name='repr-name-b')

    @ray.remote
    class Actor:
        pass
    a = Actor.remote()
    wait_for_condition(_verify_repr_name, id=a._actor_id.hex(), name='')

    @ray.remote
    class AsyncActor:

        def __init__(self, x) -> None:
            if False:
                while True:
                    i = 10
            self.x = x

        def __repr__(self) -> str:
            if False:
                print('Hello World!')
            return self.x

        async def ready(self):
            pass
    a = AsyncActor.remote(x='async-x')
    wait_for_condition(_verify_repr_name, id=a._actor_id.hex(), name='async-x')
    a = ReprActor.options(max_concurrency=3).remote(x='x')
    wait_for_condition(_verify_repr_name, id=a._actor_id.hex(), name='x')
    a = ReprActor.options(name='named-actor').remote(x='repr-name')
    wait_for_condition(_verify_repr_name, id=a._actor_id.hex(), name='repr-name')
    a = ReprActor.options(name='detached-actor', lifetime='detached').remote(x='repr-name')
    wait_for_condition(_verify_repr_name, id=a._actor_id.hex(), name='repr-name')
    ray.kill(a)

    class OutClass:

        @ray.remote
        class InnerActor:

            def __init__(self, name) -> None:
                if False:
                    i = 10
                    return i + 15
                self.name = name

            def __repr__(self) -> str:
                if False:
                    for i in range(10):
                        print('nop')
                return self.name

        def get_actor(self, name):
            if False:
                while True:
                    i = 10
            return OutClass.InnerActor.remote(name=name)
    a = OutClass().get_actor(name='inner')
    wait_for_condition(_verify_repr_name, id=a._actor_id.hex(), name='inner')

def test_experimental_import_deprecation():
    if False:
        for i in range(10):
            print('nop')
    with pytest.warns(DeprecationWarning):
        from ray.experimental.state.api import list_tasks
    with pytest.warns(DeprecationWarning):
        from ray.experimental.state.common import DEFAULT_RPC_TIMEOUT
    with pytest.warns(DeprecationWarning):
        from ray.experimental.state.custom_types import ACTOR_STATUS
    with pytest.warns(DeprecationWarning):
        from ray.experimental.state.exception import RayStateApiException
    with pytest.warns(DeprecationWarning):
        from ray.experimental.state.state_cli import ray_get
    with pytest.warns(DeprecationWarning):
        from ray.experimental.state.state_manager import StateDataSourceClient
    with pytest.warns(DeprecationWarning):
        from ray.experimental.state.util import convert_string_to_type

def test_actor_task_with_repr_name(ray_start_with_dashboard):
    if False:
        return 10

    @ray.remote
    class ReprActor:

        def __init__(self, x) -> None:
            if False:
                return 10
            self.x = x

        def __repr__(self) -> str:
            if False:
                while True:
                    i = 10
            return self.x

        def f(self):
            if False:
                while True:
                    i = 10
            pass
    a = ReprActor.remote(x='repr-name-a')
    ray.get(a.f.remote())

    def verify():
        if False:
            for i in range(10):
                print('nop')
        tasks = list_tasks(detail=True, filters=[('type', '=', 'ACTOR_TASK')])
        assert len(tasks) == 1, tasks
        assert tasks[0].name == 'repr-name-a.f'
        assert tasks[0].func_or_class_name == 'ReprActor.f'
        return True
    wait_for_condition(verify)
    b = ReprActor.remote(x='repr-name-b')
    ray.get(b.f.options(name='custom-name').remote())

    def verify():
        if False:
            return 10
        tasks = list_tasks(detail=True, filters=[('actor_id', '=', b._actor_id.hex()), ('type', '=', 'ACTOR_TASK')])
        assert len(tasks) == 1, tasks
        assert tasks[0].name == 'custom-name'
        assert tasks[0].func_or_class_name == 'ReprActor.f'
        return True
    wait_for_condition(verify)

    @ray.remote
    class Actor:

        def f(self):
            if False:
                while True:
                    i = 10
            pass
    c = Actor.remote()
    ray.get(c.f.remote())

    def verify():
        if False:
            for i in range(10):
                print('nop')
        tasks = list_tasks(detail=True, filters=[('actor_id', '=', c._actor_id.hex()), ('type', '=', 'ACTOR_TASK')])
        assert len(tasks) == 1, tasks
        assert tasks[0].name == 'Actor.f'
        assert tasks[0].func_or_class_name == 'Actor.f'
        return True
    wait_for_condition(verify)

@pytest.mark.skipif(sys.platform == 'win32', reason='Release test not expected to work on non-linux.')
def test_state_api_scale_smoke(shutdown_only):
    if False:
        print('Hello World!')
    ray.init()
    release_test_file_path = '../../release/nightly_tests/stress_tests/test_state_api_scale.py'
    full_path = Path(ray.__file__).parents[0] / release_test_file_path
    assert full_path.exists()
    check_call_subprocess(['python', str(full_path), '--smoke-test'])

def test_ray_timeline(shutdown_only):
    if False:
        return 10
    ray.init(num_cpus=8)

    @ray.remote
    def f():
        if False:
            while True:
                i = 10
        pass
    ray.get(f.remote())
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = os.path.join(tmpdirname, 'timeline.json')
        ray.timeline(filename)
        with open(filename, 'r') as f:
            dumped = json.load(f)
        assert len(dumped) > 0
if __name__ == '__main__':
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))