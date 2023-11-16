import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock, patch
import pytest
from ray.serve._private.common import DeploymentID, DeploymentInfo, DeploymentStatus, ReplicaName, ReplicaState, ReplicaTag
from ray.serve._private.config import DeploymentConfig, ReplicaConfig
from ray.serve._private.constants import DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_S, DEFAULT_GRACEFUL_SHUTDOWN_WAIT_LOOP_S, DEFAULT_HEALTH_CHECK_PERIOD_S, DEFAULT_HEALTH_CHECK_TIMEOUT_S, DEFAULT_MAX_CONCURRENT_QUERIES
from ray.serve._private.deployment_scheduler import ReplicaSchedulingRequest
from ray.serve._private.deployment_state import ActorReplicaWrapper, DeploymentReplica, DeploymentState, DeploymentStateManager, DeploymentVersion, ReplicaStartupStatus, ReplicaStateContainer, VersionedReplica
from ray.serve._private.utils import get_random_letters
from ray.serve.tests.common.utils import MockKVStore, MockTimer

class FakeRemoteFunction:

    def remote(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class MockActorHandle:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._actor_id = 'fake_id'
        self.initialize_and_get_metadata_called = False
        self.is_allocated_called = False

    @property
    def initialize_and_get_metadata(self):
        if False:
            for i in range(10):
                print('nop')
        self.initialize_and_get_metadata_called = True
        return FakeRemoteFunction()

    @property
    def is_allocated(self):
        if False:
            return 10
        self.is_allocated_called = True
        return FakeRemoteFunction()

class MockReplicaActorWrapper:

    def __init__(self, actor_name: str, controller_name: str, replica_tag: ReplicaTag, deployment_id: DeploymentID, version: DeploymentVersion):
        if False:
            return 10
        self._actor_name = actor_name
        self._replica_tag = replica_tag
        self._deployment_id = deployment_id
        self.started = False
        self.recovering = False
        self.version = version
        self.ready = ReplicaStartupStatus.PENDING_ALLOCATION
        self.stopped = False
        self.done_stopping = False
        self.force_stopped_counter = 0
        self.health_check_called = False
        self.healthy = True
        self._is_cross_language = False
        self._actor_handle = MockActorHandle()
        self._node_id = None
        self._node_id_is_set = False

    @property
    def is_cross_language(self) -> bool:
        if False:
            print('Hello World!')
        return self._is_cross_language

    @property
    def replica_tag(self) -> str:
        if False:
            i = 10
            return i + 15
        return str(self._replica_tag)

    @property
    def deployment_name(self) -> str:
        if False:
            print('Hello World!')
        return self._deployment_id.name

    @property
    def actor_handle(self) -> MockActorHandle:
        if False:
            return 10
        return self._actor_handle

    @property
    def max_concurrent_queries(self) -> int:
        if False:
            i = 10
            return i + 15
        return self.version.deployment_config.max_concurrent_queries

    @property
    def graceful_shutdown_timeout_s(self) -> float:
        if False:
            while True:
                i = 10
        return self.version.deployment_config.graceful_shutdown_timeout_s

    @property
    def health_check_period_s(self) -> float:
        if False:
            print('Hello World!')
        return self.version.deployment_config.health_check_period_s

    @property
    def health_check_timeout_s(self) -> float:
        if False:
            print('Hello World!')
        return self.version.deployment_config.health_check_timeout_s

    @property
    def pid(self) -> Optional[int]:
        if False:
            print('Hello World!')
        return None

    @property
    def actor_id(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        return None

    @property
    def worker_id(self) -> Optional[str]:
        if False:
            return 10
        return None

    @property
    def node_id(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        if self._node_id_is_set:
            return self._node_id
        if self.ready == ReplicaStartupStatus.SUCCEEDED or self.started:
            return 'node-id'
        return None

    @property
    def availability_zone(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        return None

    @property
    def node_ip(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        return None

    @property
    def log_file_path(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        return None

    def set_ready(self, version: DeploymentVersion=None):
        if False:
            return 10
        self.ready = ReplicaStartupStatus.SUCCEEDED
        if version:
            self.version_to_be_fetched_from_actor = version
        else:
            self.version_to_be_fetched_from_actor = self.version

    def set_failed_to_start(self):
        if False:
            return 10
        self.ready = ReplicaStartupStatus.FAILED

    def set_done_stopping(self):
        if False:
            return 10
        self.done_stopping = True

    def set_unhealthy(self):
        if False:
            print('Hello World!')
        self.healthy = False

    def set_starting_version(self, version: DeploymentVersion):
        if False:
            while True:
                i = 10
        'Mocked deployment_worker return version from reconfigure()'
        self.starting_version = version

    def set_node_id(self, node_id: str):
        if False:
            for i in range(10):
                print('nop')
        self._node_id = node_id
        self._node_id_is_set = True

    def start(self, deployment_info: DeploymentInfo):
        if False:
            i = 10
            return i + 15
        self.started = True
        return ReplicaSchedulingRequest(deployment_id=self._deployment_id, replica_name=self._replica_tag, actor_def=None, actor_resources=None, actor_options=None, actor_init_args=None, on_scheduled=None)

    def reconfigure(self, version: DeploymentVersion):
        if False:
            while True:
                i = 10
        self.started = True
        updating = self.version.requires_actor_reconfigure(version)
        self.version = version
        return updating

    def recover(self):
        if False:
            return 10
        self.recovering = True
        self.started = False

    def check_ready(self) -> ReplicaStartupStatus:
        if False:
            return 10
        ready = self.ready
        self.ready = ReplicaStartupStatus.PENDING_INITIALIZATION
        if ready == ReplicaStartupStatus.SUCCEEDED and self.recovering:
            self.recovering = False
            self.started = True
            self.version = self.version_to_be_fetched_from_actor
        return (ready, None)

    def resource_requirements(self) -> Tuple[str, str]:
        if False:
            print('Hello World!')
        assert self.started
        return (str({'REQUIRED_RESOURCE': 1.0}), str({'AVAILABLE_RESOURCE': 1.0}))

    @property
    def actor_resources(self) -> Dict[str, float]:
        if False:
            for i in range(10):
                print('nop')
        return {'CPU': 0.1}

    @property
    def available_resources(self) -> Dict[str, float]:
        if False:
            return 10
        return {}

    def graceful_stop(self) -> None:
        if False:
            print('Hello World!')
        assert self.started
        self.stopped = True
        return self.graceful_shutdown_timeout_s

    def check_stopped(self) -> bool:
        if False:
            return 10
        return self.done_stopping

    def force_stop(self):
        if False:
            while True:
                i = 10
        self.force_stopped_counter += 1

    def check_health(self):
        if False:
            return 10
        self.health_check_called = True
        return self.healthy

class MockDeploymentScheduler:

    def __init__(self, cluster_node_info_cache):
        if False:
            i = 10
            return i + 15
        self.deployments = set()
        self.replicas = defaultdict(set)

    def on_deployment_created(self, deployment_id, scheduling_strategy):
        if False:
            while True:
                i = 10
        assert deployment_id not in self.deployments
        self.deployments.add(deployment_id)

    def on_deployment_deleted(self, deployment_id):
        if False:
            return 10
        assert deployment_id in self.deployments
        self.deployments.remove(deployment_id)

    def on_replica_stopping(self, deployment_id, replica_name):
        if False:
            for i in range(10):
                print('nop')
        assert replica_name in self.replicas[deployment_id]
        self.replicas[deployment_id].remove(replica_name)

    def on_replica_running(self, deployment_id, replica_name, node_id):
        if False:
            while True:
                i = 10
        assert replica_name in self.replicas[deployment_id]

    def on_replica_recovering(self, deployment_id, replica_name):
        if False:
            print('Hello World!')
        assert replica_name not in self.replicas[deployment_id]
        self.replicas[deployment_id].add(replica_name)

    def schedule(self, upscales, downscales):
        if False:
            print('Hello World!')
        for upscale in upscales.values():
            for replica_scheduling_request in upscale:
                assert replica_scheduling_request.replica_name not in self.replicas[replica_scheduling_request.deployment_id]
                self.replicas[replica_scheduling_request.deployment_id].add(replica_scheduling_request.replica_name)
        deployment_to_replicas_to_stop = defaultdict(set)
        for downscale in downscales.values():
            replica_iter = iter(self.replicas[downscale.deployment_id])
            for _ in range(downscale.num_to_stop):
                deployment_to_replicas_to_stop[downscale.deployment_id].add(next(replica_iter))
        return deployment_to_replicas_to_stop

def deployment_info(version: Optional[str]=None, num_replicas: Optional[int]=1, user_config: Optional[Any]=None, **config_opts) -> Tuple[DeploymentInfo, DeploymentVersion]:
    if False:
        for i in range(10):
            print('nop')
    info = DeploymentInfo(version=version, start_time_ms=0, deployment_config=DeploymentConfig(num_replicas=num_replicas, user_config=user_config, **config_opts), replica_config=ReplicaConfig.create(lambda x: x), deployer_job_id='')
    if version is not None:
        code_version = version
    else:
        code_version = get_random_letters()
    version = DeploymentVersion(code_version, info.deployment_config, info.replica_config.ray_actor_options)
    return (info, version)

def deployment_version(code_version) -> DeploymentVersion:
    if False:
        while True:
            i = 10
    return DeploymentVersion(code_version, DeploymentConfig(), {})

class MockClusterNodeInfoCache:

    def __init__(self):
        if False:
            print('Hello World!')
        self.alive_node_ids = set()
        self.draining_node_ids = set()

    def get_alive_node_ids(self):
        if False:
            while True:
                i = 10
        return self.alive_node_ids

    def get_draining_node_ids(self):
        if False:
            for i in range(10):
                print('nop')
        return self.draining_node_ids

    def get_active_node_ids(self):
        if False:
            return 10
        return self.alive_node_ids - self.draining_node_ids

    def get_node_az(self, node_id):
        if False:
            for i in range(10):
                print('nop')
        return None

@pytest.fixture
def mock_deployment_state(request) -> Tuple[DeploymentState, Mock, Mock]:
    if False:
        return 10
    timer = MockTimer()
    with patch('ray.serve._private.deployment_state.ActorReplicaWrapper', new=MockReplicaActorWrapper), patch('time.time', new=timer.time), patch('ray.serve._private.long_poll.LongPollHost') as mock_long_poll:

        def mock_save_checkpoint_fn(*args, **kwargs):
            if False:
                print('Hello World!')
            pass
        cluster_node_info_cache = MockClusterNodeInfoCache()
        deployment_state = DeploymentState(DeploymentID('name', 'my_app'), 'name', mock_long_poll, MockDeploymentScheduler(cluster_node_info_cache), cluster_node_info_cache, mock_save_checkpoint_fn)
        yield (deployment_state, timer, cluster_node_info_cache)

def replica(version: Optional[DeploymentVersion]=None) -> VersionedReplica:
    if False:
        while True:
            i = 10
    if version is None:
        version = DeploymentVersion(get_random_letters(), DeploymentConfig(), {})

    class MockVersionedReplica(VersionedReplica):

        def __init__(self, version: DeploymentVersion):
            if False:
                print('Hello World!')
            self._version = version

        @property
        def version(self):
            if False:
                i = 10
                return i + 15
            return self._version

        def update_state(self, state):
            if False:
                while True:
                    i = 10
            pass
    return MockVersionedReplica(version)

class TestReplicaStateContainer:

    def test_count(self):
        if False:
            for i in range(10):
                print('nop')
        c = ReplicaStateContainer()
        (r1, r2, r3) = (replica(deployment_version('1')), replica(deployment_version('2')), replica(deployment_version('2')))
        c.add(ReplicaState.STARTING, r1)
        c.add(ReplicaState.STARTING, r2)
        c.add(ReplicaState.STOPPING, r3)
        assert c.count() == 3
        assert c.count() == c.count(states=[ReplicaState.STARTING, ReplicaState.STOPPING])
        assert c.count(states=[ReplicaState.STARTING]) == 2
        assert c.count(states=[ReplicaState.STOPPING]) == 1
        assert c.count(version=deployment_version('1')) == 1
        assert c.count(version=deployment_version('2')) == 2
        assert c.count(version=deployment_version('3')) == 0
        assert c.count(exclude_version=deployment_version('1')) == 2
        assert c.count(exclude_version=deployment_version('2')) == 1
        assert c.count(exclude_version=deployment_version('3')) == 3
        assert c.count(version=deployment_version('1'), states=[ReplicaState.STARTING]) == 1
        assert c.count(version=deployment_version('3'), states=[ReplicaState.STARTING]) == 0
        assert c.count(version=deployment_version('2'), states=[ReplicaState.STARTING, ReplicaState.STOPPING]) == 2
        assert c.count(exclude_version=deployment_version('1'), states=[ReplicaState.STARTING]) == 1
        assert c.count(exclude_version=deployment_version('3'), states=[ReplicaState.STARTING]) == 2
        assert c.count(exclude_version=deployment_version('2'), states=[ReplicaState.STARTING, ReplicaState.STOPPING]) == 1

    def test_get(self):
        if False:
            for i in range(10):
                print('nop')
        c = ReplicaStateContainer()
        (r1, r2, r3) = (replica(), replica(), replica())
        c.add(ReplicaState.STARTING, r1)
        c.add(ReplicaState.STARTING, r2)
        c.add(ReplicaState.STOPPING, r3)
        assert c.get() == [r1, r2, r3]
        assert c.get() == c.get([ReplicaState.STARTING, ReplicaState.STOPPING])
        assert c.get([ReplicaState.STARTING]) == [r1, r2]
        assert c.get([ReplicaState.STOPPING]) == [r3]

    def test_pop_basic(self):
        if False:
            print('Hello World!')
        c = ReplicaStateContainer()
        (r1, r2, r3) = (replica(), replica(), replica())
        c.add(ReplicaState.STARTING, r1)
        c.add(ReplicaState.STARTING, r2)
        c.add(ReplicaState.STOPPING, r3)
        assert c.pop() == [r1, r2, r3]
        assert not c.pop()

    def test_pop_exclude_version(self):
        if False:
            for i in range(10):
                print('nop')
        c = ReplicaStateContainer()
        (r1, r2, r3) = (replica(deployment_version('1')), replica(deployment_version('1')), replica(deployment_version('2')))
        c.add(ReplicaState.STARTING, r1)
        c.add(ReplicaState.STARTING, r2)
        c.add(ReplicaState.STARTING, r3)
        assert c.pop(exclude_version=deployment_version('1')) == [r3]
        assert not c.pop(exclude_version=deployment_version('1'))
        assert c.pop(exclude_version=deployment_version('2')) == [r1, r2]
        assert not c.pop(exclude_version=deployment_version('2'))
        assert not c.pop()

    def test_pop_max_replicas(self):
        if False:
            while True:
                i = 10
        c = ReplicaStateContainer()
        (r1, r2, r3) = (replica(), replica(), replica())
        c.add(ReplicaState.STARTING, r1)
        c.add(ReplicaState.STARTING, r2)
        c.add(ReplicaState.STOPPING, r3)
        assert not c.pop(max_replicas=0)
        assert len(c.pop(max_replicas=1)) == 1
        assert len(c.pop(max_replicas=2)) == 2
        c.add(ReplicaState.STARTING, r1)
        c.add(ReplicaState.STARTING, r2)
        c.add(ReplicaState.STOPPING, r3)
        assert len(c.pop(max_replicas=10)) == 3

    def test_pop_states(self):
        if False:
            for i in range(10):
                print('nop')
        c = ReplicaStateContainer()
        (r1, r2, r3, r4) = (replica(), replica(), replica(), replica())
        c.add(ReplicaState.STOPPING, r1)
        c.add(ReplicaState.STARTING, r2)
        c.add(ReplicaState.STOPPING, r3)
        assert c.pop(states=[ReplicaState.STARTING]) == [r2]
        assert not c.pop(states=[ReplicaState.STARTING])
        assert c.pop(states=[ReplicaState.STOPPING]) == [r1, r3]
        assert not c.pop(states=[ReplicaState.STOPPING])
        c.add(ReplicaState.STOPPING, r1)
        c.add(ReplicaState.STARTING, r2)
        c.add(ReplicaState.STOPPING, r3)
        c.add(ReplicaState.STARTING, r4)
        assert c.pop(states=[ReplicaState.STOPPING, ReplicaState.STARTING]) == [r1, r3, r2, r4]
        assert not c.pop(states=[ReplicaState.STOPPING, ReplicaState.STARTING])
        assert not c.pop(states=[ReplicaState.STOPPING])
        assert not c.pop(states=[ReplicaState.STARTING])
        assert not c.pop()

    def test_pop_integration(self):
        if False:
            while True:
                i = 10
        c = ReplicaStateContainer()
        (r1, r2, r3, r4) = (replica(deployment_version('1')), replica(deployment_version('2')), replica(deployment_version('2')), replica(deployment_version('3')))
        c.add(ReplicaState.STOPPING, r1)
        c.add(ReplicaState.STARTING, r2)
        c.add(ReplicaState.RUNNING, r3)
        c.add(ReplicaState.RUNNING, r4)
        assert not c.pop(exclude_version=deployment_version('1'), states=[ReplicaState.STOPPING])
        assert c.pop(exclude_version=deployment_version('1'), states=[ReplicaState.RUNNING], max_replicas=1) == [r3]
        assert c.pop(exclude_version=deployment_version('1'), states=[ReplicaState.RUNNING], max_replicas=1) == [r4]
        c.add(ReplicaState.RUNNING, r3)
        c.add(ReplicaState.RUNNING, r4)
        assert c.pop(exclude_version=deployment_version('1'), states=[ReplicaState.RUNNING]) == [r3, r4]
        assert c.pop(exclude_version=deployment_version('1'), states=[ReplicaState.STARTING]) == [r2]
        c.add(ReplicaState.STARTING, r2)
        c.add(ReplicaState.RUNNING, r3)
        c.add(ReplicaState.RUNNING, r4)
        assert c.pop(exclude_version=deployment_version('1'), states=[ReplicaState.RUNNING, ReplicaState.STARTING]) == [r3, r4, r2]
        assert c.pop(exclude_version=deployment_version('nonsense'), states=[ReplicaState.STOPPING]) == [r1]

def check_counts(deployment_state: DeploymentState, total: Optional[int]=None, version: Optional[str]=None, by_state: Optional[List[Tuple[ReplicaState, int]]]=None):
    if False:
        while True:
            i = 10
    if total is not None:
        assert deployment_state._replicas.count(version=version) == total
    if by_state is not None:
        for (state, count) in by_state:
            assert isinstance(state, ReplicaState)
            assert isinstance(count, int) and count >= 0
            curr_count = deployment_state._replicas.count(version=version, states=[state])
            msg = f'Expected {count} for state {state} but got {curr_count}.'
            assert curr_count == count, msg

@pytest.mark.parametrize('mock_deployment_state', [True, False], indirect=True)
def test_create_delete_single_replica(mock_deployment_state):
    if False:
        i = 10
        return i + 15
    (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
    cluster_node_info_cache.alive_node_ids = {'node-id'}
    (b_info_1, b_version_1) = deployment_info()
    updating = deployment_state.deploy(b_info_1)
    assert updating
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, total=1, by_state=[(ReplicaState.STARTING, 1)])
    deployment_state.update()
    check_counts(deployment_state, total=1, by_state=[(ReplicaState.STARTING, 1)])
    deployment_state._replicas.get()[0]._actor.set_ready()
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    deployment_state.update()
    check_counts(deployment_state, total=1, by_state=[(ReplicaState.RUNNING, 1)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
    deployment_state.delete()
    deployment_state_update_result = deployment_state.update()
    replicas_to_stop = deployment_state._deployment_scheduler.schedule({}, {deployment_state._id: deployment_state_update_result.downscale} if deployment_state_update_result.downscale else {})[deployment_state._id]
    deployment_state.stop_replicas(replicas_to_stop)
    check_counts(deployment_state, total=1, by_state=[(ReplicaState.STOPPING, 1)])
    assert deployment_state._replicas.get()[0]._actor.stopped
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    replica = deployment_state._replicas.get()[0]
    replica._actor.set_done_stopping()
    deployment_state_update_result = deployment_state.update()
    assert deployment_state_update_result.deleted
    check_counts(deployment_state, total=0)

@pytest.mark.parametrize('mock_deployment_state', [True, False], indirect=True)
def test_force_kill(mock_deployment_state):
    if False:
        print('Hello World!')
    (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
    cluster_node_info_cache.alive_node_ids = {'node-id'}
    grace_period_s = 10
    (b_info_1, b_version_1) = deployment_info(graceful_shutdown_timeout_s=grace_period_s)
    deployment_state.deploy(b_info_1)
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    deployment_state._replicas.get()[0]._actor.set_ready()
    deployment_state.update()
    deployment_state.delete()
    deployment_state_update_result = deployment_state.update()
    replicas_to_stop = deployment_state._deployment_scheduler.schedule({}, {deployment_state._id: deployment_state_update_result.downscale} if deployment_state_update_result.downscale else {})[deployment_state._id]
    deployment_state.stop_replicas(replicas_to_stop)
    check_counts(deployment_state, total=1, by_state=[(ReplicaState.STOPPING, 1)])
    assert deployment_state._replicas.get()[0]._actor.stopped
    for _ in range(10):
        deployment_state.update()
    assert not deployment_state._replicas.get()[0]._actor.force_stopped_counter
    check_counts(deployment_state, total=1, by_state=[(ReplicaState.STOPPING, 1)])
    timer.advance(grace_period_s + 0.1)
    deployment_state.update()
    assert deployment_state._replicas.get()[0]._actor.force_stopped_counter == 1
    check_counts(deployment_state, total=1, by_state=[(ReplicaState.STOPPING, 1)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    deployment_state.update()
    assert deployment_state._replicas.get()[0]._actor.force_stopped_counter == 2
    check_counts(deployment_state, total=1, by_state=[(ReplicaState.STOPPING, 1)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    replica = deployment_state._replicas.get()[0]
    replica._actor.set_done_stopping()
    deployment_state_update_result = deployment_state.update()
    assert deployment_state_update_result.deleted
    check_counts(deployment_state, total=0)

@pytest.mark.parametrize('mock_deployment_state', [True, False], indirect=True)
def test_redeploy_same_version(mock_deployment_state):
    if False:
        for i in range(10):
            print('nop')
    (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
    cluster_node_info_cache.alive_node_ids = {'node-id'}
    (b_info_1, b_version_1) = deployment_info(version='1')
    updating = deployment_state.deploy(b_info_1)
    assert updating
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, version=b_version_1, total=1, by_state=[(ReplicaState.STARTING, 1)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    updating = deployment_state.deploy(b_info_1)
    assert not updating
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    deployment_state.update()
    check_counts(deployment_state, version=b_version_1, total=1, by_state=[(ReplicaState.STARTING, 1)])
    deployment_state._replicas.get()[0]._actor.set_ready()
    deployment_state.update()
    check_counts(deployment_state, version=b_version_1, total=1, by_state=[(ReplicaState.RUNNING, 1)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
    updating = deployment_state.deploy(b_info_1)
    assert not updating
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
    check_counts(deployment_state, version=b_version_1, total=1, by_state=[(ReplicaState.RUNNING, 1)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY

@pytest.mark.parametrize('mock_deployment_state', [True, False], indirect=True)
def test_redeploy_no_version(mock_deployment_state):
    if False:
        return 10
    (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
    cluster_node_info_cache.alive_node_ids = {'node-id'}
    (b_info_1, b_version_1) = deployment_info(version=None)
    updating = deployment_state.deploy(b_info_1)
    assert updating
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, total=1, by_state=[(ReplicaState.STARTING, 1)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    updating = deployment_state.deploy(b_info_1)
    assert updating
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    deployment_state.update()
    check_counts(deployment_state, total=1, by_state=[(ReplicaState.STOPPING, 1)])
    deployment_state.update()
    deployment_state._replicas.get(states=[ReplicaState.STOPPING])[0]._actor.set_done_stopping()
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, total=1, by_state=[(ReplicaState.STARTING, 1)])
    deployment_state._replicas.get(states=[ReplicaState.STARTING])[0]._actor.set_ready()
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    deployment_state.update()
    check_counts(deployment_state, total=1)
    check_counts(deployment_state, total=1, by_state=[(ReplicaState.RUNNING, 1)])
    deployment_state.update()
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
    (b_info_3, b_version_3) = deployment_info(version='3')
    updating = deployment_state.deploy(b_info_3)
    assert updating
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    deployment_state.update()
    check_counts(deployment_state, total=1)
    check_counts(deployment_state, total=1, by_state=[(ReplicaState.STOPPING, 1)])
    deployment_state.update()
    deployment_state._replicas.get(states=[ReplicaState.STOPPING])[0]._actor.set_done_stopping()
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    deployment_state._replicas.get(states=[ReplicaState.STARTING])[0]._actor.set_ready()
    check_counts(deployment_state, total=1, by_state=[(ReplicaState.STARTING, 1)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    deployment_state_update_result = deployment_state.update()
    assert not deployment_state_update_result.deleted
    check_counts(deployment_state, total=1, by_state=[(ReplicaState.RUNNING, 1)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY

@pytest.mark.parametrize('mock_deployment_state', [True, False], indirect=True)
def test_redeploy_new_version(mock_deployment_state):
    if False:
        for i in range(10):
            print('nop')
    (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
    cluster_node_info_cache.alive_node_ids = {'node-id'}
    (b_info_1, b_version_1) = deployment_info(version='1')
    updating = deployment_state.deploy(b_info_1)
    assert updating
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, version=b_version_1, total=1, by_state=[(ReplicaState.STARTING, 1)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    (b_info_2, b_version_2) = deployment_info(version='2')
    updating = deployment_state.deploy(b_info_2)
    assert updating
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    deployment_state.update()
    check_counts(deployment_state, version=b_version_1, total=1, by_state=[(ReplicaState.STOPPING, 1)])
    deployment_state.update()
    deployment_state._replicas.get(states=[ReplicaState.STOPPING])[0]._actor.set_done_stopping()
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, version=b_version_2, total=1, by_state=[(ReplicaState.STARTING, 1)])
    deployment_state._replicas.get(states=[ReplicaState.STARTING])[0]._actor.set_ready()
    deployment_state.update()
    check_counts(deployment_state, total=1)
    check_counts(deployment_state, version=b_version_2, total=1, by_state=[(ReplicaState.RUNNING, 1)])
    deployment_state.update()
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
    (b_info_3, b_version_3) = deployment_info(version='3')
    updating = deployment_state.deploy(b_info_3)
    assert updating
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    deployment_state.update()
    check_counts(deployment_state, total=1)
    check_counts(deployment_state, version=b_version_2, total=1, by_state=[(ReplicaState.STOPPING, 1)])
    deployment_state.update()
    deployment_state._replicas.get(states=[ReplicaState.STOPPING])[0]._actor.set_done_stopping()
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    deployment_state._replicas.get(states=[ReplicaState.STARTING])[0]._actor.set_ready()
    check_counts(deployment_state, version=b_version_3, total=1, by_state=[(ReplicaState.STARTING, 1)])
    deployment_state_update_result = deployment_state.update()
    assert not deployment_state_update_result.deleted
    check_counts(deployment_state, version=b_version_3, total=1, by_state=[(ReplicaState.RUNNING, 1)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY

@pytest.mark.parametrize('mock_deployment_state', [True, False], indirect=True)
@pytest.mark.parametrize('option,value', [('user_config', {'hello': 'world'}), ('max_concurrent_queries', 10), ('graceful_shutdown_timeout_s', DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_S + 1), ('graceful_shutdown_wait_loop_s', DEFAULT_GRACEFUL_SHUTDOWN_WAIT_LOOP_S + 1), ('health_check_period_s', DEFAULT_HEALTH_CHECK_PERIOD_S + 1), ('health_check_timeout_s', DEFAULT_HEALTH_CHECK_TIMEOUT_S + 1)])
def test_deploy_new_config_same_code_version(mock_deployment_state, option, value):
    if False:
        while True:
            i = 10
    (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
    cluster_node_info_cache.alive_node_ids = {'node-id'}
    (b_info_1, b_version_1) = deployment_info(version='1')
    updated = deployment_state.deploy(b_info_1)
    assert updated
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    deployment_state._replicas.get()[0]._actor.set_ready()
    deployment_state.update()
    check_counts(deployment_state, version=b_version_1, total=1, by_state=[(ReplicaState.RUNNING, 1)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
    (b_info_2, b_version_2) = deployment_info(version='1', **{option: value})
    updated = deployment_state.deploy(b_info_2)
    assert updated
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    check_counts(deployment_state, version=b_version_1, total=1, by_state=[(ReplicaState.RUNNING, 1)])
    if option in ['user_config', 'graceful_shutdown_wait_loop_s']:
        deployment_state.update()
        check_counts(deployment_state, total=1)
        check_counts(deployment_state, version=b_version_2, total=1, by_state=[(ReplicaState.UPDATING, 1)])
        deployment_state._replicas.get()[0]._actor.set_ready()
    deployment_state.update()
    check_counts(deployment_state, total=1)
    check_counts(deployment_state, version=b_version_2, total=1, by_state=[(ReplicaState.RUNNING, 1)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY

@pytest.mark.parametrize('mock_deployment_state', [True, False], indirect=True)
def test_deploy_new_config_same_code_version_2(mock_deployment_state):
    if False:
        print('Hello World!')
    (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
    cluster_node_info_cache.alive_node_ids = {'node-id'}
    (b_info_1, b_version_1) = deployment_info(version='1')
    updated = deployment_state.deploy(b_info_1)
    assert updated
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, version=b_version_1, total=1, by_state=[(ReplicaState.STARTING, 1)])
    (b_info_2, b_version_2) = deployment_info(version='1', user_config={'hello': 'world'})
    updated = deployment_state.deploy(b_info_2)
    assert updated
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    deployment_state.update()
    check_counts(deployment_state, version=b_version_1, total=1, by_state=[(ReplicaState.STARTING, 1)])
    deployment_state._replicas.get()[0]._actor.set_ready()
    deployment_state.update()
    check_counts(deployment_state, version=b_version_2, total=1, by_state=[(ReplicaState.UPDATING, 1)])
    deployment_state._replicas.get()[0]._actor.set_ready()
    deployment_state.update()
    check_counts(deployment_state, total=1)
    check_counts(deployment_state, version=b_version_2, total=1, by_state=[(ReplicaState.RUNNING, 1)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY

@pytest.mark.parametrize('mock_deployment_state', [True, False], indirect=True)
def test_deploy_new_config_new_version(mock_deployment_state):
    if False:
        i = 10
        return i + 15
    (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
    cluster_node_info_cache.alive_node_ids = {'node-id'}
    (b_info_1, b_version_1) = deployment_info(version='1')
    updating = deployment_state.deploy(b_info_1)
    assert updating
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    deployment_state._replicas.get()[0]._actor.set_ready()
    deployment_state.update()
    check_counts(deployment_state, version=b_version_1, total=1, by_state=[(ReplicaState.RUNNING, 1)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
    (b_info_2, b_version_2) = deployment_info(version='2', user_config={'hello': 'world'})
    updating = deployment_state.deploy(b_info_2)
    assert updating
    deployment_state.update()
    check_counts(deployment_state, version=b_version_1, total=1, by_state=[(ReplicaState.STOPPING, 1)])
    deployment_state._replicas.get(states=[ReplicaState.STOPPING])[0]._actor.set_done_stopping()
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    deployment_state._replicas.get(states=[ReplicaState.STARTING])[0]._actor.set_ready()
    check_counts(deployment_state, version=b_version_2, total=1, by_state=[(ReplicaState.STARTING, 1)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    deployment_state.update()
    check_counts(deployment_state, version=b_version_2, total=1, by_state=[(ReplicaState.RUNNING, 1)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY

@pytest.mark.parametrize('mock_deployment_state', [True, False], indirect=True)
def test_stop_replicas_on_draining_nodes(mock_deployment_state):
    if False:
        for i in range(10):
            print('nop')
    (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
    cluster_node_info_cache.alive_node_ids = {'node-1', 'node-2'}
    (b_info_1, b_version_1) = deployment_info(num_replicas=2, version='1')
    updating = deployment_state.deploy(b_info_1)
    assert updating
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.STARTING, 2)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    cluster_node_info_cache.draining_node_ids = {'node-2'}
    deployment_state.update()
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.STARTING, 2)])
    deployment_state._replicas.get()[0]._actor.set_ready()
    deployment_state._replicas.get()[0]._actor.set_node_id('node-1')
    deployment_state._replicas.get()[1]._actor.set_ready()
    deployment_state._replicas.get()[1]._actor.set_node_id('node-2')
    deployment_state.update()
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.RUNNING, 1), (ReplicaState.STOPPING, 1)])
    cluster_node_info_cache.alive_node_ids = {'node-1', 'node-2', 'node-3'}
    deployment_state._replicas.get()[1]._actor.set_done_stopping()
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.STARTING, 1), (ReplicaState.RUNNING, 1)])

@pytest.mark.parametrize('mock_deployment_state', [True, False], indirect=True)
def test_initial_deploy_no_throttling(mock_deployment_state):
    if False:
        while True:
            i = 10
    (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
    cluster_node_info_cache.alive_node_ids = {str(i) for i in range(10)}
    (b_info_1, b_version_1) = deployment_info(num_replicas=10, version='1')
    updating = deployment_state.deploy(b_info_1)
    assert updating
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, total=10, by_state=[(ReplicaState.STARTING, 10)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    for replica in deployment_state._replicas.get():
        replica._actor.set_ready()
    deployment_state.update()
    check_counts(deployment_state, total=10, by_state=[(ReplicaState.RUNNING, 10)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY

@pytest.mark.parametrize('mock_deployment_state', [True, False], indirect=True)
def test_new_version_deploy_throttling(mock_deployment_state):
    if False:
        return 10
    (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
    cluster_node_info_cache.alive_node_ids = {str(i) for i in range(10)}
    (b_info_1, b_version_1) = deployment_info(num_replicas=10, version='1', user_config='1')
    updating = deployment_state.deploy(b_info_1)
    assert updating
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, total=10, by_state=[(ReplicaState.STARTING, 10)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    for replica in deployment_state._replicas.get():
        replica._actor.set_ready()
    deployment_state.update()
    check_counts(deployment_state, total=10, by_state=[(ReplicaState.RUNNING, 10)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
    (b_info_2, b_version_2) = deployment_info(num_replicas=10, version='2', user_config='2')
    updating = deployment_state.deploy(b_info_2)
    assert updating
    deployment_state.update()
    check_counts(deployment_state, version=b_version_1, total=10, by_state=[(ReplicaState.RUNNING, 8), (ReplicaState.STOPPING, 2)])
    deployment_state._replicas.get(states=[ReplicaState.STOPPING])[0]._actor.set_done_stopping()
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, total=10)
    check_counts(deployment_state, version=b_version_1, total=9, by_state=[(ReplicaState.RUNNING, 8), (ReplicaState.STOPPING, 1)])
    check_counts(deployment_state, version=b_version_2, total=1, by_state=[(ReplicaState.STARTING, 1)])
    deployment_state._replicas.get(states=[ReplicaState.STARTING])[0]._actor.set_ready()
    deployment_state.update()
    check_counts(deployment_state, total=10)
    check_counts(deployment_state, version=b_version_1, total=9, by_state=[(ReplicaState.RUNNING, 7), (ReplicaState.STOPPING, 2)])
    check_counts(deployment_state, version=b_version_2, total=1, by_state=[(ReplicaState.RUNNING, 1)])
    deployment_state._replicas.get(states=[ReplicaState.STOPPING])[0]._actor.set_done_stopping()
    deployment_state._replicas.get(states=[ReplicaState.STOPPING])[1]._actor.set_done_stopping()
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    new_replicas = 1
    old_replicas = 9
    while old_replicas > 3:
        deployment_state_update_result = deployment_state.update()
        deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
        check_counts(deployment_state, total=10)
        check_counts(deployment_state, version=b_version_1, total=old_replicas - 2, by_state=[(ReplicaState.RUNNING, old_replicas - 2)])
        check_counts(deployment_state, version=b_version_2, total=new_replicas + 2, by_state=[(ReplicaState.RUNNING, new_replicas), (ReplicaState.STARTING, 2)])
        deployment_state._replicas.get(states=[ReplicaState.STARTING])[0]._actor.set_ready()
        deployment_state._replicas.get(states=[ReplicaState.STARTING])[1]._actor.set_ready()
        new_replicas += 2
        old_replicas -= 2
        deployment_state.update()
        check_counts(deployment_state, total=10)
        check_counts(deployment_state, version=b_version_1, total=old_replicas, by_state=[(ReplicaState.RUNNING, old_replicas - 2), (ReplicaState.STOPPING, 2)])
        check_counts(deployment_state, version=b_version_2, total=new_replicas, by_state=[(ReplicaState.RUNNING, new_replicas)])
        deployment_state._replicas.get(states=[ReplicaState.STOPPING])[0]._actor.set_done_stopping()
        deployment_state._replicas.get(states=[ReplicaState.STOPPING])[1]._actor.set_done_stopping()
        assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, total=10)
    check_counts(deployment_state, version=b_version_1, total=1, by_state=[(ReplicaState.RUNNING, 1)])
    check_counts(deployment_state, version=b_version_2, total=9, by_state=[(ReplicaState.RUNNING, 7), (ReplicaState.STARTING, 2)])
    deployment_state._replicas.get(states=[ReplicaState.STARTING])[0]._actor.set_ready()
    deployment_state._replicas.get(states=[ReplicaState.STARTING])[1]._actor.set_ready()
    deployment_state.update()
    check_counts(deployment_state, total=10)
    check_counts(deployment_state, version=b_version_1, total=1, by_state=[(ReplicaState.STOPPING, 1)])
    check_counts(deployment_state, version=b_version_2, total=9, by_state=[(ReplicaState.RUNNING, 9)])
    deployment_state._replicas.get(states=[ReplicaState.STOPPING])[0]._actor.set_done_stopping()
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, total=10)
    check_counts(deployment_state, version=b_version_2, total=10, by_state=[(ReplicaState.RUNNING, 9), (ReplicaState.STARTING, 1)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    deployment_state._replicas.get(states=[ReplicaState.STARTING])[0]._actor.set_ready()
    deployment_state.update()
    check_counts(deployment_state, total=10)
    check_counts(deployment_state, version=b_version_2, total=10, by_state=[(ReplicaState.RUNNING, 10)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY

@pytest.mark.parametrize('mock_deployment_state', [True, False], indirect=True)
def test_reconfigure_throttling(mock_deployment_state):
    if False:
        i = 10
        return i + 15
    (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
    cluster_node_info_cache.alive_node_ids = {str(i) for i in range(2)}
    (b_info_1, b_version_1) = deployment_info(num_replicas=2, version='1', user_config='1')
    updating = deployment_state.deploy(b_info_1)
    assert updating
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.STARTING, 2)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    for replica in deployment_state._replicas.get():
        replica._actor.set_ready()
    deployment_state.update()
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.RUNNING, 2)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
    (b_info_2, b_version_2) = deployment_info(num_replicas=2, version='1', user_config='2')
    updating = deployment_state.deploy(b_info_2)
    assert updating
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    deployment_state.update()
    check_counts(deployment_state, version=b_version_1, total=1, by_state=[(ReplicaState.RUNNING, 1)])
    check_counts(deployment_state, version=b_version_2, total=1, by_state=[(ReplicaState.UPDATING, 1)])
    deployment_state._replicas.get(states=[ReplicaState.UPDATING])[0]._actor.set_ready()
    deployment_state.update()
    check_counts(deployment_state, version=b_version_2, total=2, by_state=[(ReplicaState.RUNNING, 1), (ReplicaState.UPDATING, 1)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    deployment_state._replicas.get(states=[ReplicaState.UPDATING])[0]._actor.set_ready()
    deployment_state.update()
    check_counts(deployment_state, version=b_version_2, total=2, by_state=[(ReplicaState.RUNNING, 2)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY

@pytest.mark.parametrize('mock_deployment_state', [False], indirect=True)
def test_new_version_and_scale_down(mock_deployment_state):
    if False:
        i = 10
        return i + 15
    (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
    cluster_node_info_cache.alive_node_ids = {'node-id'}
    (b_info_1, b_version_1) = deployment_info(num_replicas=10, version='1')
    updating = deployment_state.deploy(b_info_1)
    assert updating
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, total=10, by_state=[(ReplicaState.STARTING, 10)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    for replica in deployment_state._replicas.get():
        replica._actor.set_ready()
    deployment_state.update()
    check_counts(deployment_state, total=10, by_state=[(ReplicaState.RUNNING, 10)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
    (b_info_2, b_version_2) = deployment_info(num_replicas=2, version='2')
    updating = deployment_state.deploy(b_info_2)
    assert updating
    deployment_state_update_result = deployment_state.update()
    replicas_to_stop = deployment_state._deployment_scheduler.schedule({}, {deployment_state._id: deployment_state_update_result.downscale})[deployment_state._id]
    deployment_state.stop_replicas(replicas_to_stop)
    check_counts(deployment_state, version=b_version_1, total=10, by_state=[(ReplicaState.RUNNING, 2), (ReplicaState.STOPPING, 8)])
    deployment_state._replicas.get(states=[ReplicaState.STOPPING])[0]._actor.set_done_stopping()
    deployment_state.update()
    check_counts(deployment_state, total=9)
    check_counts(deployment_state, version=b_version_1, total=9, by_state=[(ReplicaState.RUNNING, 2), (ReplicaState.STOPPING, 7)])
    for replica in deployment_state._replicas.get(states=[ReplicaState.STOPPING]):
        replica._actor.set_done_stopping()
    deployment_state.update()
    check_counts(deployment_state, total=2)
    check_counts(deployment_state, version=b_version_1, total=2, by_state=[(ReplicaState.RUNNING, 1), (ReplicaState.STOPPING, 1)])
    deployment_state._replicas.get(states=[ReplicaState.STOPPING])[0]._actor.set_done_stopping()
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, total=2)
    check_counts(deployment_state, version=b_version_1, total=1, by_state=[(ReplicaState.RUNNING, 1)])
    check_counts(deployment_state, version=b_version_2, total=1, by_state=[(ReplicaState.STARTING, 1)])
    deployment_state._replicas.get(states=[ReplicaState.STARTING])[0]._actor.set_ready()
    deployment_state.update()
    check_counts(deployment_state, total=2)
    check_counts(deployment_state, version=b_version_1, total=1, by_state=[(ReplicaState.STOPPING, 1)])
    check_counts(deployment_state, version=b_version_2, total=1, by_state=[(ReplicaState.RUNNING, 1)])
    deployment_state._replicas.get(states=[ReplicaState.STOPPING])[0]._actor.set_done_stopping()
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, total=2)
    check_counts(deployment_state, version=b_version_2, total=2, by_state=[(ReplicaState.RUNNING, 1), (ReplicaState.STARTING, 1)])
    deployment_state._replicas.get(states=[ReplicaState.STARTING])[0]._actor.set_ready()
    deployment_state.update()
    check_counts(deployment_state, total=2)
    check_counts(deployment_state, version=b_version_2, total=2, by_state=[(ReplicaState.RUNNING, 2)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY

@pytest.mark.parametrize('mock_deployment_state', [False], indirect=True)
def test_new_version_and_scale_up(mock_deployment_state):
    if False:
        i = 10
        return i + 15
    (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
    (b_info_1, b_version_1) = deployment_info(num_replicas=2, version='1')
    updating = deployment_state.deploy(b_info_1)
    assert updating
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.STARTING, 2)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    for replica in deployment_state._replicas.get():
        replica._actor.set_ready()
    deployment_state.update()
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.RUNNING, 2)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
    (b_info_2, b_version_2) = deployment_info(num_replicas=10, version='2')
    updating = deployment_state.deploy(b_info_2)
    assert updating
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, version=b_version_1, total=2, by_state=[(ReplicaState.RUNNING, 2)])
    check_counts(deployment_state, version=b_version_2, total=8, by_state=[(ReplicaState.STARTING, 8)])
    for replica in deployment_state._replicas.get(states=[ReplicaState.STARTING]):
        replica._actor.set_ready()
    deployment_state.update()
    check_counts(deployment_state, version=b_version_1, total=2, by_state=[(ReplicaState.RUNNING, 0), (ReplicaState.STOPPING, 2)])
    check_counts(deployment_state, version=b_version_2, total=8, by_state=[(ReplicaState.RUNNING, 8)])
    for replica in deployment_state._replicas.get(states=[ReplicaState.STOPPING]):
        replica._actor.set_done_stopping()
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, total=10)
    check_counts(deployment_state, version=b_version_2, total=10, by_state=[(ReplicaState.RUNNING, 8), (ReplicaState.STARTING, 2)])
    for replica in deployment_state._replicas.get(states=[ReplicaState.STARTING]):
        replica._actor.set_ready()
    deployment_state.update()
    check_counts(deployment_state, total=10)
    check_counts(deployment_state, version=b_version_2, total=10, by_state=[(ReplicaState.RUNNING, 10)])
    deployment_state.update()
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY

@pytest.mark.parametrize('mock_deployment_state', [True, False], indirect=True)
def test_health_check(mock_deployment_state):
    if False:
        i = 10
        return i + 15
    (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
    cluster_node_info_cache.alive_node_ids = {str(i) for i in range(2)}
    (b_info_1, b_version_1) = deployment_info(num_replicas=2, version='1')
    updating = deployment_state.deploy(b_info_1)
    assert updating
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.STARTING, 2)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    for replica in deployment_state._replicas.get():
        replica._actor.set_ready()
        assert not replica._actor.health_check_called
    deployment_state.update()
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.RUNNING, 2)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
    deployment_state.update()
    for replica in deployment_state._replicas.get():
        assert replica._actor.health_check_called
    deployment_state._replicas.get()[0]._actor.set_unhealthy()
    deployment_state.update()
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.RUNNING, 1), (ReplicaState.STOPPING, 1)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.UNHEALTHY
    replica = deployment_state._replicas.get(states=[ReplicaState.STOPPING])[0]
    replica._actor.set_done_stopping()
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.RUNNING, 1), (ReplicaState.STARTING, 1)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.UNHEALTHY
    replica = deployment_state._replicas.get(states=[ReplicaState.STARTING])[0]
    replica._actor.set_ready()
    assert deployment_state.curr_status_info.status == DeploymentStatus.UNHEALTHY
    deployment_state.update()
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.RUNNING, 2)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY

@pytest.mark.parametrize('mock_deployment_state', [True, False], indirect=True)
def test_update_while_unhealthy(mock_deployment_state):
    if False:
        return 10
    (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
    cluster_node_info_cache.alive_node_ids = {str(i) for i in range(2)}
    (b_info_1, b_version_1) = deployment_info(num_replicas=2, version='1')
    updating = deployment_state.deploy(b_info_1)
    assert updating
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.STARTING, 2)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    for replica in deployment_state._replicas.get():
        replica._actor.set_ready()
        assert not replica._actor.health_check_called
    deployment_state.update()
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.RUNNING, 2)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
    deployment_state.update()
    for replica in deployment_state._replicas.get():
        assert replica._actor.health_check_called
    deployment_state._replicas.get()[0]._actor.set_unhealthy()
    deployment_state.update()
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.RUNNING, 1), (ReplicaState.STOPPING, 1)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.UNHEALTHY
    replica = deployment_state._replicas.get(states=[ReplicaState.STOPPING])[0]
    replica._actor.set_done_stopping()
    (b_info_2, b_version_2) = deployment_info(num_replicas=2, version='2')
    updating = deployment_state.deploy(b_info_2)
    assert updating
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, version=b_version_1, total=1, by_state=[(ReplicaState.RUNNING, 1)])
    check_counts(deployment_state, version=b_version_2, total=1, by_state=[(ReplicaState.STARTING, 1)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    deployment_state._replicas.get(states=[ReplicaState.RUNNING])[0]._actor.set_unhealthy()
    deployment_state.update()
    check_counts(deployment_state, version=b_version_1, total=1, by_state=[(ReplicaState.STOPPING, 1)])
    check_counts(deployment_state, version=b_version_2, total=1, by_state=[(ReplicaState.STARTING, 1)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    replica = deployment_state._replicas.get(states=[ReplicaState.STOPPING])[0]
    replica._actor.set_done_stopping()
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, version=b_version_2, total=2, by_state=[(ReplicaState.STARTING, 2)])
    for replica in deployment_state._replicas.get(states=[ReplicaState.STARTING]):
        replica._actor.set_ready()
    deployment_state.update()
    check_counts(deployment_state, version=b_version_2, total=2, by_state=[(ReplicaState.RUNNING, 2)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY

def _constructor_failure_loop_two_replica(deployment_state, num_loops):
    if False:
        for i in range(10):
            print('nop')
    'Helper function to exact constructor failure loops.'
    for i in range(num_loops):
        deployment_state_update_result = deployment_state.update()
        deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
        check_counts(deployment_state, total=2, by_state=[(ReplicaState.STARTING, 2)])
        assert deployment_state._replica_constructor_retry_counter == i * 2
        replica_1 = deployment_state._replicas.get()[0]
        replica_2 = deployment_state._replicas.get()[1]
        replica_1._actor.set_failed_to_start()
        replica_2._actor.set_failed_to_start()
        deployment_state.update()
        check_counts(deployment_state, total=2, by_state=[(ReplicaState.STOPPING, 2)])
        replica_1._actor.set_done_stopping()
        replica_2._actor.set_done_stopping()

@pytest.mark.parametrize('mock_deployment_state', [True, False], indirect=True)
def test_deploy_with_consistent_constructor_failure(mock_deployment_state):
    if False:
        i = 10
        return i + 15
    '\n    Test deploy() multiple replicas with consistent constructor failure.\n\n    The deployment should get marked FAILED.\n    '
    (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
    cluster_node_info_cache.alive_node_ids = {str(i) for i in range(2)}
    (b_info_1, b_version_1) = deployment_info(num_replicas=2)
    updating = deployment_state.deploy(b_info_1)
    assert updating
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    _constructor_failure_loop_two_replica(deployment_state, 3)
    assert deployment_state._replica_constructor_retry_counter == 6
    assert deployment_state.curr_status_info.status == DeploymentStatus.UNHEALTHY
    check_counts(deployment_state, total=2)
    assert deployment_state.curr_status_info.message != ''

@pytest.mark.parametrize('mock_deployment_state', [True, False], indirect=True)
def test_deploy_with_partial_constructor_failure(mock_deployment_state):
    if False:
        while True:
            i = 10
    "\n    Test deploy() multiple replicas with constructor failure exceedining\n    pre-set limit but achieved partial success with at least 1 running replica.\n\n    Ensures:\n        1) Deployment status doesn't get marked FAILED.\n        2) There should be expected # of RUNNING replicas eventually that\n            matches user intent\n        3) Replica counter set as -1 to stop tracking current goal as it's\n            already completed\n\n    Same testing for same test case in test_deploy.py.\n    "
    (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
    cluster_node_info_cache.alive_node_ids = {str(i) for i in range(2)}
    (b_info_1, b_version_1) = deployment_info(num_replicas=2)
    updating = deployment_state.deploy(b_info_1)
    assert updating
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    _constructor_failure_loop_two_replica(deployment_state, 2)
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.STARTING, 2)])
    assert deployment_state._replica_constructor_retry_counter == 4
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    replica_1 = deployment_state._replicas.get()[0]
    replica_2 = deployment_state._replicas.get()[1]
    replica_1._actor.set_ready()
    replica_2._actor.set_failed_to_start()
    deployment_state.update()
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.RUNNING, 1)])
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.STOPPING, 1)])
    deployment_state.update()
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.RUNNING, 1)])
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.STOPPING, 1)])
    replica_2._actor.set_done_stopping()
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.RUNNING, 1)])
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.STARTING, 1)])
    starting_replica = deployment_state._replicas.get(states=[ReplicaState.STARTING])[0]
    starting_replica._actor.set_failed_to_start()
    deployment_state.update()
    assert deployment_state._replica_constructor_retry_counter == -1
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.RUNNING, 1)])
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.STOPPING, 1)])
    deployment_state.update()
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.RUNNING, 1)])
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.STOPPING, 1)])
    starting_replica = deployment_state._replicas.get(states=[ReplicaState.STOPPING])[0]
    starting_replica._actor.set_done_stopping()
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.RUNNING, 1)])
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.STARTING, 1)])
    starting_replica = deployment_state._replicas.get(states=[ReplicaState.STARTING])[0]
    starting_replica._actor.set_ready()
    deployment_state.update()
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.RUNNING, 2)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY

@pytest.mark.parametrize('mock_deployment_state', [True, False], indirect=True)
def test_deploy_with_transient_constructor_failure(mock_deployment_state):
    if False:
        while True:
            i = 10
    "\n    Test deploy() multiple replicas with transient constructor failure.\n    Ensures:\n        1) Deployment status gets marked as RUNNING.\n        2) There should be expected # of RUNNING replicas eventually that\n            matches user intent.\n        3) Replica counter set as -1 to stop tracking current goal as it's\n            already completed.\n\n    Same testing for same test case in test_deploy.py.\n    "
    (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
    cluster_node_info_cache.alive_node_ids = {str(i) for i in range(2)}
    (b_info_1, b_version_1) = deployment_info(num_replicas=2)
    updating = deployment_state.deploy(b_info_1)
    assert updating
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    _constructor_failure_loop_two_replica(deployment_state, 2)
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    deployment_state_update_result = deployment_state.update()
    deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.STARTING, 2)])
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    assert deployment_state._replica_constructor_retry_counter == 4
    replica_1 = deployment_state._replicas.get()[0]
    replica_2 = deployment_state._replicas.get()[1]
    replica_1._actor.set_ready()
    replica_2._actor.set_ready()
    deployment_state.update()
    check_counts(deployment_state, total=2, by_state=[(ReplicaState.RUNNING, 2)])
    assert deployment_state._replica_constructor_retry_counter == 4
    assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY

@pytest.mark.parametrize('mock_deployment_state', [False], indirect=True)
def test_exponential_backoff(mock_deployment_state):
    if False:
        i = 10
        return i + 15
    'Test exponential backoff.'
    (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
    cluster_node_info_cache.alive_node_ids = {str(i) for i in range(2)}
    (b_info_1, b_version_1) = deployment_info(num_replicas=2)
    updating = deployment_state.deploy(b_info_1)
    assert updating
    assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
    _constructor_failure_loop_two_replica(deployment_state, 3)
    assert deployment_state._replica_constructor_retry_counter == 6
    last_retry = timer.time()
    for i in range(7):
        while timer.time() - last_retry < 2 ** i:
            deployment_state.update()
            assert deployment_state._replica_constructor_retry_counter == 6 + 2 * i
            check_counts(deployment_state, total=0)
            timer.advance(0.1)
        timer.advance(5)
        check_counts(deployment_state, total=0)
        deployment_state_update_result = deployment_state.update()
        deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
        last_retry = timer.time()
        check_counts(deployment_state, total=2)
        replica_1 = deployment_state._replicas.get()[0]
        replica_2 = deployment_state._replicas.get()[1]
        replica_1._actor.set_failed_to_start()
        replica_2._actor.set_failed_to_start()
        timer.advance(0.1)
        deployment_state.update()
        check_counts(deployment_state, total=2, by_state=[(ReplicaState.STOPPING, 2)])
        timer.advance(0.1)
        replica_1._actor.set_done_stopping()
        replica_2._actor.set_done_stopping()
        deployment_state.update()
        check_counts(deployment_state, total=0)
        timer.advance(0.1)

@pytest.fixture
def mock_deployment_state_manager_full(request) -> Tuple[DeploymentStateManager, Mock, Mock]:
    if False:
        for i in range(10):
            print('nop')
    "Fully mocked deployment state manager.\n\n    i.e kv store and gcs client is mocked so we don't need to initialize\n    ray. Also, since this is used for some recovery tests, this yields a\n    method for creating a new mocked deployment state manager.\n    "
    timer = MockTimer()
    with patch('ray.serve._private.deployment_state.ActorReplicaWrapper', new=MockReplicaActorWrapper), patch('ray.serve._private.default_impl.create_deployment_scheduler') as mock_create_deployment_scheduler, patch('time.time', new=timer.time), patch('ray.serve._private.long_poll.LongPollHost') as mock_long_poll, patch('ray.get_runtime_context'):
        kv_store = MockKVStore()
        cluster_node_info_cache = MockClusterNodeInfoCache()

        def create_deployment_state_manager(actor_names=None, placement_group_names=None):
            if False:
                for i in range(10):
                    print('nop')
            if actor_names is None:
                actor_names = []
            if placement_group_names is None:
                placement_group_names = []
            mock_create_deployment_scheduler.return_value = MockDeploymentScheduler(cluster_node_info_cache)
            return DeploymentStateManager('name', kv_store, mock_long_poll, actor_names, placement_group_names, cluster_node_info_cache)
        yield (create_deployment_state_manager, timer, cluster_node_info_cache)

def test_recover_state_from_replica_names(mock_deployment_state_manager_full):
    if False:
        for i in range(10):
            print('nop')
    'Test recover deployment state.'
    deployment_id = DeploymentID('test_deployment', 'test_app')
    (create_deployment_state_manager, _, cluster_node_info_cache) = mock_deployment_state_manager_full
    deployment_state_manager = create_deployment_state_manager()
    cluster_node_info_cache.alive_node_ids = {'node-id'}
    (info1, version1) = deployment_info(version='1')
    updating = deployment_state_manager.deploy(deployment_id, info1)
    deployment_state = deployment_state_manager._deployment_states[deployment_id]
    assert updating
    deployment_state_manager.update()
    check_counts(deployment_state, total=1, version=version1, by_state=[(ReplicaState.STARTING, 1)])
    mocked_replica = deployment_state._replicas.get()[0]
    mocked_replica._actor.set_ready()
    deployment_state_manager.update()
    check_counts(deployment_state, total=1, version=version1, by_state=[(ReplicaState.RUNNING, 1)])
    new_deployment_state_manager = create_deployment_state_manager([ReplicaName.prefix + mocked_replica.replica_tag])
    new_deployment_state = new_deployment_state_manager._deployment_states[deployment_id]
    check_counts(new_deployment_state, total=1, version=version1, by_state=[(ReplicaState.RECOVERING, 1)])
    new_mocked_replica = new_deployment_state._replicas.get()[0]
    new_mocked_replica._actor.set_ready(version1)
    any_recovering = new_deployment_state_manager.update()
    check_counts(new_deployment_state, total=1, version=version1, by_state=[(ReplicaState.RUNNING, 1)])
    assert not any_recovering
    assert mocked_replica.replica_tag == new_mocked_replica.replica_tag

def test_recover_during_rolling_update(mock_deployment_state_manager_full):
    if False:
        for i in range(10):
            print('nop')
    'Test controller crashes before a replica is updated to new version.\n\n    During recovery, the controller should wait for the version to be fetched from\n    the replica actor. Once it is fetched and the controller realizes the replica\n    has an outdated version, it should be stopped and a new replica should be started\n    with the target version.\n    '
    deployment_id = DeploymentID('test_deployment', 'test_app')
    (create_deployment_state_manager, _, cluster_node_info_cache) = mock_deployment_state_manager_full
    deployment_state_manager = create_deployment_state_manager()
    cluster_node_info_cache.alive_node_ids = {'node-id'}
    (info1, version1) = deployment_info(version='1')
    updating = deployment_state_manager.deploy(deployment_id, info1)
    deployment_state = deployment_state_manager._deployment_states[deployment_id]
    assert updating
    deployment_state_manager.update()
    check_counts(deployment_state, total=1, version=version1, by_state=[(ReplicaState.STARTING, 1)])
    mocked_replica = deployment_state._replicas.get()[0]
    mocked_replica._actor.set_ready()
    deployment_state_manager.update()
    check_counts(deployment_state, total=1, version=version1, by_state=[(ReplicaState.RUNNING, 1)])
    (info2, version2) = deployment_info(version='2')
    updating = deployment_state_manager.deploy(deployment_id, info2)
    assert updating
    new_deployment_state_manager = create_deployment_state_manager([ReplicaName.prefix + mocked_replica.replica_tag])
    cluster_node_info_cache.alive_node_ids = {'node-id'}
    new_deployment_state = new_deployment_state_manager._deployment_states[deployment_id]
    check_counts(new_deployment_state, total=1, version=version2, by_state=[(ReplicaState.RECOVERING, 1)])
    for _ in range(3):
        new_deployment_state_manager.update()
        check_counts(new_deployment_state, total=1, version=version2, by_state=[(ReplicaState.RECOVERING, 1)])
    new_mocked_replica = new_deployment_state._replicas.get()[0]
    new_mocked_replica._actor.set_ready(version1)
    new_deployment_state_manager.update()
    new_deployment_state_manager.update()
    check_counts(new_deployment_state, total=1, version=version1, by_state=[(ReplicaState.STOPPING, 1)])
    new_mocked_replica._actor.set_done_stopping()
    new_deployment_state_manager.update()
    check_counts(new_deployment_state, total=1, version=version2, by_state=[(ReplicaState.STARTING, 1)])
    new_mocked_replica_version2 = new_deployment_state._replicas.get()[0]
    new_mocked_replica_version2._actor.set_ready()
    new_deployment_state_manager.update()
    check_counts(new_deployment_state, total=1, version=version2, by_state=[(ReplicaState.RUNNING, 1)])
    assert mocked_replica.replica_tag != new_mocked_replica_version2.replica_tag

@pytest.fixture
def mock_deployment_state_manager(request) -> Tuple[DeploymentStateManager, Mock, Mock]:
    if False:
        while True:
            i = 10
    timer = MockTimer()
    with patch('ray.serve._private.deployment_state.ActorReplicaWrapper', new=MockReplicaActorWrapper), patch('ray.serve._private.default_impl.create_deployment_scheduler') as mock_create_deployment_scheduler, patch('time.time', new=timer.time), patch('ray.serve._private.long_poll.LongPollHost') as mock_long_poll:
        kv_store = MockKVStore()
        cluster_node_info_cache = MockClusterNodeInfoCache()
        mock_create_deployment_scheduler.return_value = MockDeploymentScheduler(cluster_node_info_cache)
        all_current_actor_names = []
        all_current_placement_group_names = []
        deployment_state_manager = DeploymentStateManager(DeploymentID('name', 'my_app'), kv_store, mock_long_poll, all_current_actor_names, all_current_placement_group_names, cluster_node_info_cache)
        yield (deployment_state_manager, timer, cluster_node_info_cache)

def test_shutdown(mock_deployment_state_manager):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that shutdown waits for all deployments to be deleted and they\n    are force-killed without a grace period.\n    '
    (deployment_state_manager, timer, cluster_node_info_cache) = mock_deployment_state_manager
    cluster_node_info_cache.alive_node_ids = {'node-id'}
    deployment_id = DeploymentID('test_deployment', 'test_app')
    grace_period_s = 10
    (b_info_1, _) = deployment_info(graceful_shutdown_timeout_s=grace_period_s)
    updating = deployment_state_manager.deploy(deployment_id, b_info_1)
    assert updating
    deployment_state = deployment_state_manager._deployment_states[deployment_id]
    deployment_state_manager.update()
    check_counts(deployment_state, total=1, by_state=[(ReplicaState.STARTING, 1)])
    deployment_state._replicas.get()[0]._actor.set_ready()
    deployment_state_manager.update()
    check_counts(deployment_state, total=1, by_state=[(ReplicaState.RUNNING, 1)])
    assert not deployment_state._replicas.get()[0]._actor.stopped
    assert not deployment_state_manager.is_ready_for_shutdown()
    deployment_state_manager.shutdown()
    timer.advance(grace_period_s + 0.1)
    deployment_state_manager.update()
    check_counts(deployment_state, total=1, by_state=[(ReplicaState.STOPPING, 1)])
    assert deployment_state._replicas.get()[0]._actor.stopped
    assert len(deployment_state_manager.get_deployment_statuses()) > 0
    replica = deployment_state._replicas.get()[0]
    replica._actor.set_done_stopping()
    deployment_state_manager.update()
    check_counts(deployment_state, total=0)
    assert len(deployment_state_manager.get_deployment_statuses()) == 0
    assert deployment_state_manager.is_ready_for_shutdown()

def test_resource_requirements_none():
    if False:
        while True:
            i = 10
    "Ensure resource_requirements doesn't break if a requirement is None"

    class FakeActor:
        actor_resources = {'num_cpus': 2, 'fake': None}
        placement_group_bundles = None
        available_resources = {}
    replica = DeploymentReplica(None, 'random_tag', None, None)
    replica._actor = FakeActor()
    replica.resource_requirements()

class TestActorReplicaWrapper:

    def test_default_value(self):
        if False:
            while True:
                i = 10
        actor_replica = ActorReplicaWrapper(version=deployment_version('1'), actor_name='test', controller_name='test_controller', replica_tag='test_tag', deployment_id=DeploymentID('test_deployment', 'test_app'))
        assert actor_replica.graceful_shutdown_timeout_s == DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_S
        assert actor_replica.max_concurrent_queries == DEFAULT_MAX_CONCURRENT_QUERIES
        assert actor_replica.health_check_period_s == DEFAULT_HEALTH_CHECK_PERIOD_S
        assert actor_replica.health_check_timeout_s == DEFAULT_HEALTH_CHECK_TIMEOUT_S

def test_get_active_node_ids(mock_deployment_state_manager_full):
    if False:
        for i in range(10):
            print('nop')
    'Test get_active_node_ids() are collecting the correct node ids\n\n    When there are no running replicas, both methods should return empty results. When\n    the replicas are in the RUNNING state, get_running_replica_node_ids() should return\n    a list of all node ids. `get_active_node_ids()` should return a set\n    of all node ids.\n    '
    node_ids = ('node1', 'node2', 'node2')
    deployment_id = DeploymentID('test_deployment', 'test_app')
    (create_deployment_state_manager, _, cluster_node_info_cache) = mock_deployment_state_manager_full
    deployment_state_manager = create_deployment_state_manager()
    cluster_node_info_cache.alive_node_ids = set(node_ids)
    (info1, version1) = deployment_info(version='1', num_replicas=3)
    updating = deployment_state_manager.deploy(deployment_id, info1)
    deployment_state = deployment_state_manager._deployment_states[deployment_id]
    assert updating
    deployment_state_manager.update()
    check_counts(deployment_state, total=3, version=version1, by_state=[(ReplicaState.STARTING, 3)])
    mocked_replicas = deployment_state._replicas.get()
    for (idx, mocked_replica) in enumerate(mocked_replicas):
        mocked_replica._actor.set_node_id(node_ids[idx])
    assert deployment_state.get_active_node_ids() == set(node_ids)
    assert deployment_state_manager.get_active_node_ids() == set(node_ids)
    for mocked_replica in mocked_replicas:
        mocked_replica._actor.set_ready()
    deployment_state_manager.update()
    check_counts(deployment_state, total=3, version=version1, by_state=[(ReplicaState.RUNNING, 3)])
    assert deployment_state.get_active_node_ids() == set(node_ids)
    assert deployment_state_manager.get_active_node_ids() == set(node_ids)
    for _ in mocked_replicas:
        deployment_state._stop_one_running_replica_for_testing()
    deployment_state_manager.update()
    check_counts(deployment_state, total=3, version=version1, by_state=[(ReplicaState.STOPPING, 3)])
    assert deployment_state.get_active_node_ids() == set()
    assert deployment_state_manager.get_active_node_ids() == set()

def test_get_active_node_ids_none(mock_deployment_state_manager_full):
    if False:
        print('Hello World!')
    'Test get_active_node_ids() are not collecting none node ids.\n\n    When the running replicas has None as the node id, `get_active_node_ids()` should\n    not include it in the set.\n    '
    node_ids = ('node1', 'node2', 'node2')
    deployment_id = DeploymentID('test_deployment', 'test_app')
    (create_deployment_state_manager, _, cluster_node_info_cache) = mock_deployment_state_manager_full
    deployment_state_manager = create_deployment_state_manager()
    cluster_node_info_cache.alive_node_ids = set(node_ids)
    (info1, version1) = deployment_info(version='1', num_replicas=3)
    updating = deployment_state_manager.deploy(deployment_id, info1)
    deployment_state = deployment_state_manager._deployment_states[deployment_id]
    assert updating
    deployment_state_manager.update()
    check_counts(deployment_state, total=3, version=version1, by_state=[(ReplicaState.STARTING, 3)])
    mocked_replicas = deployment_state._replicas.get()
    for (idx, mocked_replica) in enumerate(mocked_replicas):
        mocked_replica._actor.set_node_id(node_ids[idx])
    assert deployment_state.get_active_node_ids() == set(node_ids)
    assert deployment_state_manager.get_active_node_ids() == set(node_ids)
    for mocked_replica in mocked_replicas:
        mocked_replica._actor.set_node_id(None)
        mocked_replica._actor.set_ready()
    deployment_state_manager.update()
    check_counts(deployment_state, total=3, version=version1, by_state=[(ReplicaState.RUNNING, 3)])
    assert None not in deployment_state.get_active_node_ids()
    assert None not in deployment_state_manager.get_active_node_ids()

class TestTargetCapacity:
    """
    Tests related to the `target_capacity` field that adjusts the target num_replicas.
    """

    @pytest.mark.parametrize('num_replicas,target_capacity,expected_output', [(10, None, 10), (10, 100, 10), (10, 99, 10), (10, 50, 5), (10, 0, 1), (10, 25, 3), (1, None, 1), (1, 100, 1), (1, 0, 1), (1, 23, 1), (3, 20, 1), (3, 40, 1), (3, 70, 2), (3, 90, 3), (0, None, 0), (0, 1, 0), (0, 99, 0), (0, 100, 0)])
    def test_get_capacity_adjusted_num_replicas(self, num_replicas: int, target_capacity: Optional[float], expected_output: int):
        if False:
            while True:
                i = 10
        result = DeploymentState.get_capacity_adjusted_num_replicas(num_replicas, target_capacity)
        assert isinstance(result, int)
        assert result == expected_output

    def test_initial_deploy(self, mock_deployment_state):
        if False:
            while True:
                i = 10
        '\n        Deploy with target_capacity set, should apply immediately.\n        '
        (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
        (b_info_1, b_version_1) = deployment_info(num_replicas=2)
        updating = deployment_state.deploy(b_info_1)
        assert updating
        deployment_state_update_result = deployment_state.update(target_capacity=50)
        deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
        check_counts(deployment_state, total=1, by_state=[(ReplicaState.STARTING, 1)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
        for replica in deployment_state._replicas.get():
            replica._actor.set_ready()
        deployment_state.update(target_capacity=50)
        check_counts(deployment_state, total=1, by_state=[(ReplicaState.RUNNING, 1)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        deployment_state.update(target_capacity=50)
        check_counts(deployment_state, total=1, by_state=[(ReplicaState.RUNNING, 1)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY

    def test_target_capacity_100_no_effect(self, mock_deployment_state):
        if False:
            for i in range(10):
                print('nop')
        '\n        Deploy with no target_capacity set, then set to 100. Should take no effect.\n\n        Then go back to no target_capacity, should still have no effect.\n        '
        (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
        (b_info_1, b_version_1) = deployment_info(num_replicas=2)
        updating = deployment_state.deploy(b_info_1)
        assert updating
        deployment_state_update_result = deployment_state.update()
        deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
        check_counts(deployment_state, total=2, by_state=[(ReplicaState.STARTING, 2)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
        for replica in deployment_state._replicas.get():
            replica._actor.set_ready()
        deployment_state.update()
        check_counts(deployment_state, total=2, by_state=[(ReplicaState.RUNNING, 2)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        deployment_state.update()
        check_counts(deployment_state, total=2, by_state=[(ReplicaState.RUNNING, 2)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        deployment_state.update(target_capacity=100)
        check_counts(deployment_state, total=2, by_state=[(ReplicaState.RUNNING, 2)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        deployment_state.update(target_capacity=None)
        check_counts(deployment_state, total=2, by_state=[(ReplicaState.RUNNING, 2)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY

    def test_target_capacity_0_1_replica(self, mock_deployment_state):
        if False:
            i = 10
            return i + 15
        '\n        Deploy with target_capacity set to 0. Should have a single replica.\n        '
        (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
        (b_info_1, b_version_1) = deployment_info(num_replicas=100)
        updating = deployment_state.deploy(b_info_1)
        assert updating
        deployment_state_update_result = deployment_state.update(target_capacity=0)
        deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
        check_counts(deployment_state, total=1, by_state=[(ReplicaState.STARTING, 1)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
        for replica in deployment_state._replicas.get():
            replica._actor.set_ready()
        deployment_state.update(target_capacity=0)
        check_counts(deployment_state, total=1, by_state=[(ReplicaState.RUNNING, 1)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        deployment_state.update(target_capacity=0)
        check_counts(deployment_state, total=1, by_state=[(ReplicaState.RUNNING, 1)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY

    def test_reduce_target_capacity(self, mock_deployment_state):
        if False:
            while True:
                i = 10
        '\n        Deploy with target capacity set to 100, then reduce to 50, then reduce to 0.\n        '
        (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
        (b_info_1, b_version_1) = deployment_info(num_replicas=10)
        updating = deployment_state.deploy(b_info_1)
        assert updating
        deployment_state_update_result = deployment_state.update(target_capacity=100)
        deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
        check_counts(deployment_state, total=10, by_state=[(ReplicaState.STARTING, 10)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
        for replica in deployment_state._replicas.get():
            replica._actor.set_ready()
        deployment_state.update(target_capacity=100)
        check_counts(deployment_state, total=10, by_state=[(ReplicaState.RUNNING, 10)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        deployment_state_update_result = deployment_state.update(target_capacity=50)
        replicas_to_stop = deployment_state._deployment_scheduler.schedule({}, {deployment_state._id: deployment_state_update_result.downscale})[deployment_state._id]
        deployment_state.stop_replicas(replicas_to_stop)
        check_counts(deployment_state, total=10, by_state=[(ReplicaState.RUNNING, 5), (ReplicaState.STOPPING, 5)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        for replica in deployment_state._replicas.get([ReplicaState.STOPPING]):
            replica._actor.set_done_stopping()
        deployment_state.update(target_capacity=50)
        check_counts(deployment_state, total=5, by_state=[(ReplicaState.RUNNING, 5)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        deployment_state_update_result = deployment_state.update(target_capacity=0)
        replicas_to_stop = deployment_state._deployment_scheduler.schedule({}, {deployment_state._id: deployment_state_update_result.downscale})[deployment_state._id]
        deployment_state.stop_replicas(replicas_to_stop)
        check_counts(deployment_state, total=5, by_state=[(ReplicaState.RUNNING, 1), (ReplicaState.STOPPING, 4)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        for replica in deployment_state._replicas.get([ReplicaState.STOPPING]):
            replica._actor.set_done_stopping()
        deployment_state.update(target_capacity=0)
        check_counts(deployment_state, total=1, by_state=[(ReplicaState.RUNNING, 1)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY

    def test_increase_target_capacity(self, mock_deployment_state):
        if False:
            while True:
                i = 10
        '\n        Deploy with target_capacity set to 0, then increase to 50, then increase to 100.\n        '
        (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
        (b_info_1, b_version_1) = deployment_info(num_replicas=10)
        updating = deployment_state.deploy(b_info_1)
        assert updating
        deployment_state_update_result = deployment_state.update(target_capacity=0)
        deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
        check_counts(deployment_state, total=1, by_state=[(ReplicaState.STARTING, 1)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
        for replica in deployment_state._replicas.get():
            replica._actor.set_ready()
        deployment_state.update(target_capacity=0)
        check_counts(deployment_state, total=1, by_state=[(ReplicaState.RUNNING, 1)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        deployment_state_update_result = deployment_state.update(target_capacity=50)
        deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
        check_counts(deployment_state, total=5, by_state=[(ReplicaState.RUNNING, 1), (ReplicaState.STARTING, 4)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        for replica in deployment_state._replicas.get():
            replica._actor.set_ready()
        deployment_state.update(target_capacity=50)
        check_counts(deployment_state, total=5, by_state=[(ReplicaState.RUNNING, 5)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        deployment_state_update_result = deployment_state.update(target_capacity=100)
        deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
        check_counts(deployment_state, total=10, by_state=[(ReplicaState.RUNNING, 5), (ReplicaState.STARTING, 5)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        for replica in deployment_state._replicas.get():
            replica._actor.set_ready()
        deployment_state.update(target_capacity=100)
        check_counts(deployment_state, total=10, by_state=[(ReplicaState.RUNNING, 10)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY

    def test_clear_target_capacity(self, mock_deployment_state):
        if False:
            i = 10
            return i + 15
        '\n        Deploy with target_capacity set, should apply immediately.\n        '
        (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
        (b_info_1, b_version_1) = deployment_info(num_replicas=10)
        updating = deployment_state.deploy(b_info_1)
        assert updating
        deployment_state_update_result = deployment_state.update(target_capacity=50)
        deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
        check_counts(deployment_state, total=5, by_state=[(ReplicaState.STARTING, 5)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
        for replica in deployment_state._replicas.get():
            replica._actor.set_ready()
        deployment_state.update(target_capacity=50)
        check_counts(deployment_state, total=5, by_state=[(ReplicaState.RUNNING, 5)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        deployment_state_update_result = deployment_state.update()
        deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
        check_counts(deployment_state, total=10, by_state=[(ReplicaState.RUNNING, 5), (ReplicaState.STARTING, 5)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        for replica in deployment_state._replicas.get():
            replica._actor.set_ready()
        deployment_state.update()
        check_counts(deployment_state, total=10, by_state=[(ReplicaState.RUNNING, 10)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY

    def test_target_num_replicas_is_zero(self, mock_deployment_state):
        if False:
            return 10
        "\n        If the target `num_replicas` is zero (i.e., scale-to-zero is enabled and it's\n        autoscaled down), then replicas should remain at zero regardless of\n        target_capacity.\n        "
        (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
        (b_info_1, b_version_1) = deployment_info(num_replicas=1)
        updating = deployment_state.deploy(b_info_1)
        assert updating
        deployment_state_update_result = deployment_state.update(target_capacity=50)
        deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
        check_counts(deployment_state, total=1, by_state=[(ReplicaState.STARTING, 1)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
        for replica in deployment_state._replicas.get():
            replica._actor.set_ready()
        deployment_state.update()
        check_counts(deployment_state, total=1, by_state=[(ReplicaState.RUNNING, 1)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        deployment_state._target_state.num_replicas = 0
        deployment_state_update_result = deployment_state.update(target_capacity=50)
        replicas_to_stop = deployment_state._deployment_scheduler.schedule({}, {deployment_state._id: deployment_state_update_result.downscale})[deployment_state._id]
        deployment_state.stop_replicas(replicas_to_stop)
        check_counts(deployment_state, total=1, by_state=[(ReplicaState.STOPPING, 1)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        for replica in deployment_state._replicas.get([ReplicaState.STOPPING]):
            replica._actor.set_done_stopping()
        deployment_state.update(target_capacity=50)
        check_counts(deployment_state, total=0)
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        deployment_state_update_result = deployment_state.update()
        assert not deployment_state_update_result.upscale
        assert not deployment_state_update_result.downscale
        check_counts(deployment_state, total=0)
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        deployment_state_update_result = deployment_state.update(target_capacity=0)
        assert not deployment_state_update_result.upscale
        assert not deployment_state_update_result.downscale
        check_counts(deployment_state, total=0)
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        deployment_state_update_result = deployment_state.update(target_capacity=50)
        assert not deployment_state_update_result.upscale
        assert not deployment_state_update_result.downscale
        check_counts(deployment_state, total=0)
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        deployment_state_update_result = deployment_state.update(target_capacity=100)
        assert not deployment_state_update_result.upscale
        assert not deployment_state_update_result.downscale
        check_counts(deployment_state, total=0)
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        deployment_state._target_state.num_replicas = 1
        deployment_state_update_result = deployment_state.update()
        deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
        check_counts(deployment_state, total=1, by_state=[(ReplicaState.STARTING, 1)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        for replica in deployment_state._replicas.get():
            replica._actor.set_ready()
        deployment_state.update()
        check_counts(deployment_state, total=1, by_state=[(ReplicaState.RUNNING, 1)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY

    def test_target_capacity_with_changing_num_replicas(self, mock_deployment_state):
        if False:
            print('Hello World!')
        '\n        Test that target_capacity works with changing num_replicas (emulating\n        autoscaling).\n        '
        (deployment_state, timer, cluster_node_info_cache) = mock_deployment_state
        (b_info_1, b_version_1) = deployment_info(num_replicas=2)
        updating = deployment_state.deploy(b_info_1)
        assert updating
        deployment_state_update_result = deployment_state.update(target_capacity=0)
        deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
        check_counts(deployment_state, total=1, by_state=[(ReplicaState.STARTING, 1)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.UPDATING
        for replica in deployment_state._replicas.get():
            replica._actor.set_ready()
        deployment_state.update(target_capacity=0)
        check_counts(deployment_state, total=1, by_state=[(ReplicaState.RUNNING, 1)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        deployment_state._target_state.num_replicas = 10
        deployment_state.update(target_capacity=0)
        check_counts(deployment_state, total=1, by_state=[(ReplicaState.RUNNING, 1)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        deployment_state_update_result = deployment_state.update(target_capacity=50)
        deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
        check_counts(deployment_state, total=5, by_state=[(ReplicaState.RUNNING, 1), (ReplicaState.STARTING, 4)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        for replica in deployment_state._replicas.get():
            replica._actor.set_ready()
        deployment_state.update(target_capacity=50)
        check_counts(deployment_state, total=5, by_state=[(ReplicaState.RUNNING, 5)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        deployment_state._target_state.num_replicas = 5
        deployment_state_update_result = deployment_state.update()
        deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
        check_counts(deployment_state, total=5, by_state=[(ReplicaState.RUNNING, 5)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        deployment_state.update(target_capacity=50)
        check_counts(deployment_state, total=5, by_state=[(ReplicaState.RUNNING, 5)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        deployment_state._target_state.num_replicas = 6
        deployment_state_update_result = deployment_state.update(target_capacity=50)
        replicas_to_stop = deployment_state._deployment_scheduler.schedule({}, {deployment_state._id: deployment_state_update_result.downscale})[deployment_state._id]
        deployment_state.stop_replicas(replicas_to_stop)
        check_counts(deployment_state, total=5, by_state=[(ReplicaState.RUNNING, 3), (ReplicaState.STOPPING, 2)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        for replica in deployment_state._replicas.get([ReplicaState.STOPPING]):
            replica._actor.set_done_stopping()
        deployment_state.update(target_capacity=50)
        check_counts(deployment_state, total=3, by_state=[(ReplicaState.RUNNING, 3)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        deployment_state_update_result = deployment_state.update()
        deployment_state._deployment_scheduler.schedule({deployment_state._id: deployment_state_update_result.upscale}, {})
        check_counts(deployment_state, total=6, by_state=[(ReplicaState.RUNNING, 3), (ReplicaState.STARTING, 3)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
        for replica in deployment_state._replicas.get():
            replica._actor.set_ready()
        deployment_state.update()
        check_counts(deployment_state, total=6, by_state=[(ReplicaState.RUNNING, 6)])
        assert deployment_state.curr_status_info.status == DeploymentStatus.HEALTHY
if __name__ == '__main__':
    sys.exit(pytest.main(['-v', '-s', __file__]))