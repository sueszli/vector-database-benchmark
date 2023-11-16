import json
import time
from typing import List, Tuple
from unittest.mock import patch
import pytest
from ray._private.test_utils import wait_for_condition
from ray.serve._private.cluster_node_info_cache import ClusterNodeInfoCache
from ray.serve._private.common import ProxyStatus
from ray.serve._private.constants import PROXY_HEALTH_CHECK_UNHEALTHY_THRESHOLD, SERVE_CONTROLLER_NAME
from ray.serve._private.proxy_state import ProxyState, ProxyStateManager, ProxyWrapper, ProxyWrapperCallStatus
from ray.serve._private.utils import Timer
from ray.serve.config import DeploymentMode, HTTPOptions
from ray.serve.schema import LoggingConfig
from ray.serve.tests.common.utils import MockTimer
HEAD_NODE_ID = 'node_id-index-head'

class MockClusterNodeInfoCache:

    def __init__(self):
        if False:
            return 10
        self.alive_nodes = []

    def get_alive_nodes(self):
        if False:
            print('Hello World!')
        return self.alive_nodes

    def get_alive_node_ids(self):
        if False:
            return 10
        return {node_id for (node_id, _) in self.alive_nodes}

class FakeProxyActor:

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        pass

    def ready(self):
        if False:
            return 10
        return json.dumps(['mock_worker_id', 'mock_log_file_path'])

    def check_health(self):
        if False:
            while True:
                i = 10
        pass

class FakeProxyWrapper(ProxyWrapper):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.actor_handle = FakeProxyActor(*args, **kwargs)
        self.ready = ProxyWrapperCallStatus.FINISHED_SUCCEED
        self.health = ProxyWrapperCallStatus.FINISHED_SUCCEED
        self.worker_id = 'mock_worker_id'
        self.log_file_path = 'mock_log_file_path'
        self.health_check_ongoing = False
        self.is_draining = False
        self.shutdown = False
        self.num_health_checks = 0

    @property
    def actor_id(self) -> str:
        if False:
            while True:
                i = 10
        pass

    def reset_health_check(self):
        if False:
            return 10
        pass

    def start_new_ready_check(self):
        if False:
            return 10
        pass

    def start_new_health_check(self):
        if False:
            i = 10
            return i + 15
        self.health_check_ongoing = True

    def start_new_drained_check(self):
        if False:
            i = 10
            return i + 15
        pass

    def is_ready(self) -> ProxyWrapperCallStatus:
        if False:
            i = 10
            return i + 15
        return self.ready

    def is_healthy(self) -> ProxyWrapperCallStatus:
        if False:
            i = 10
            return i + 15
        self.num_health_checks += 1
        self.health_check_ongoing = False
        return self.health

    def is_drained(self) -> ProxyWrapperCallStatus:
        if False:
            i = 10
            return i + 15
        pass

    def is_shutdown(self):
        if False:
            for i in range(10):
                print('nop')
        return self.shutdown

    def update_draining(self, draining: bool):
        if False:
            i = 10
            return i + 15
        pass

    def kill(self):
        if False:
            for i in range(10):
                print('nop')
        self.shutdown = True

    def get_num_health_checks(self):
        if False:
            print('Hello World!')
        return self.num_health_checks

def _create_proxy_state_manager(http_options: HTTPOptions=HTTPOptions(), head_node_id: str=HEAD_NODE_ID, cluster_node_info_cache=MockClusterNodeInfoCache(), actor_proxy_wrapper_class=FakeProxyWrapper, timer=Timer()) -> (ProxyStateManager, ClusterNodeInfoCache):
    if False:
        return 10
    return (ProxyStateManager(SERVE_CONTROLLER_NAME, config=http_options, head_node_id=head_node_id, cluster_node_info_cache=cluster_node_info_cache, logging_config=LoggingConfig(), actor_proxy_wrapper_class=actor_proxy_wrapper_class, timer=timer), cluster_node_info_cache)

def _create_proxy_state(actor_proxy_wrapper_class=FakeProxyWrapper, status: ProxyStatus=ProxyStatus.STARTING, node_id: str='mock_node_id', timer=Timer(), **kwargs) -> ProxyState:
    if False:
        print('Hello World!')
    state = ProxyState(actor_proxy_wrapper=actor_proxy_wrapper_class(), actor_name='alice', node_id=node_id, node_ip='mock_node_ip', timer=timer)
    state.set_status(status=status)
    return state

@pytest.fixture
def number_of_worker_nodes() -> int:
    if False:
        print('Hello World!')
    return 100

@pytest.fixture
def all_nodes(number_of_worker_nodes) -> List[Tuple[str, str]]:
    if False:
        while True:
            i = 10
    return [(HEAD_NODE_ID, 'fake-head-ip')] + [(f'worker-node-id-{i}', f'fake-worker-ip-{i}') for i in range(number_of_worker_nodes)]

def _update_and_check_proxy_status(state: ProxyState, status: ProxyStatus):
    if False:
        i = 10
        return i + 15
    state.update()
    assert state.status == status, state.status
    return True

def _update_and_check_proxy_state_manager(proxy_state_manager: ProxyStateManager, node_ids: List[str], statuses: List[ProxyStatus], **kwargs):
    if False:
        for i in range(10):
            print('nop')
    proxy_state_manager.update(**kwargs)
    proxy_states = proxy_state_manager._proxy_states
    assert all([proxy_states[node_ids[idx]].status == statuses[idx] for idx in range(len(node_ids))]), [proxy_state.status for proxy_state in proxy_states.values()]
    return True

def test_node_selection(all_nodes):
    if False:
        for i in range(10):
            print('nop')
    all_node_ids = {node_id for (node_id, _) in all_nodes}
    (proxy_state_manager, cluster_node_info_cache) = _create_proxy_state_manager(HTTPOptions(location=DeploymentMode.NoServer))
    cluster_node_info_cache.alive_nodes = all_nodes
    assert proxy_state_manager._get_target_nodes(all_node_ids) == []
    (proxy_state_manager, cluster_node_info_cache) = _create_proxy_state_manager(HTTPOptions(location=DeploymentMode.HeadOnly))
    cluster_node_info_cache.alive_nodes = all_nodes
    assert proxy_state_manager._get_target_nodes(all_node_ids) == all_nodes[:1]
    (proxy_state_manager, cluster_node_info_cache) = _create_proxy_state_manager(HTTPOptions(location=DeploymentMode.EveryNode))
    cluster_node_info_cache.alive_nodes = all_nodes
    assert proxy_state_manager._get_target_nodes(all_node_ids) == all_nodes
    (proxy_state_manager, cluster_node_info_cache) = _create_proxy_state_manager(HTTPOptions(location=DeploymentMode.EveryNode))
    cluster_node_info_cache.alive_nodes = all_nodes
    assert proxy_state_manager._get_target_nodes({HEAD_NODE_ID}) == [(HEAD_NODE_ID, 'fake-head-ip')]

def test_proxy_state_update_restarts_unhealthy_proxies(all_nodes):
    if False:
        i = 10
        return i + 15
    'Test the update method in ProxyStateManager would\n       kill and restart unhealthy proxies.\n\n    Set up a ProxyState with UNHEALTHY status. Calls the update method on the\n    ProxyStateManager object. Expects the unhealthy proxy being replaced\n    by a new proxy with STARTING status.\n    The unhealthy proxy state is also shutting down.\n    '
    (proxy_state_manager, cluster_node_info_cache) = _create_proxy_state_manager()
    cluster_node_info_cache.alive_nodes = all_nodes
    proxy_state_manager.update()
    old_proxy_state = proxy_state_manager._proxy_states[HEAD_NODE_ID]
    old_proxy = old_proxy_state.actor_handle
    old_proxy_state.set_status(ProxyStatus.UNHEALTHY)
    wait_for_condition(condition_predictor=_update_and_check_proxy_state_manager, proxy_state_manager=proxy_state_manager, node_ids=[HEAD_NODE_ID], statuses=[ProxyStatus.HEALTHY])
    new_proxy = proxy_state_manager._proxy_states[HEAD_NODE_ID].actor_handle
    assert old_proxy_state._shutting_down
    assert new_proxy != old_proxy

def test_proxy_state_update_shutting_down():
    if False:
        while True:
            i = 10
    'Test calling update method on ProxyState when the proxy state is shutting\n    down.\n\n    This should be no-op. The status of the http proxy state will not be changed.\n    '
    proxy_state = _create_proxy_state()
    previous_status = proxy_state.status
    proxy_state.shutdown()
    proxy_state.update()
    current_status = proxy_state.status
    assert proxy_state._shutting_down
    assert previous_status == current_status

def test_proxy_state_update_starting_ready_succeed():
    if False:
        print('Hello World!')
    'Test calling update method on ProxyState when the proxy state is STARTING and\n    when the ready call succeeded.\n\n    The proxy state started with STARTING. After update is called and ready call\n    succeeded, the state will change to HEALTHY.\n    '
    proxy_state = _create_proxy_state()
    assert proxy_state.status == ProxyStatus.STARTING
    assert proxy_state.actor_details.worker_id is None
    assert proxy_state.actor_details.log_file_path is None
    assert proxy_state.actor_details.status == ProxyStatus.STARTING.value
    wait_for_condition(condition_predictor=_update_and_check_proxy_status, state=proxy_state, status=ProxyStatus.HEALTHY)
    assert proxy_state.actor_details.worker_id == 'mock_worker_id'
    assert proxy_state.actor_details.log_file_path == 'mock_log_file_path'
    assert proxy_state.actor_details.status == ProxyStatus.HEALTHY.value

def test_proxy_state_update_starting_ready_failed_once():
    if False:
        for i in range(10):
            print('nop')
    'Test calling update method on ProxyState when the proxy state is STARTING and\n    when the ready call failed once and succeeded for the following call.\n\n    The proxy state started with STARTING status. After update is called for the first\n    time and read call is blocked, the status is not changed to UNHEALTHY immediately\n    and should stay as STARTING. The following ready call is unblocked and succeed. The\n    status will then change to HEALTHY.\n    '
    proxy_state = _create_proxy_state()
    assert proxy_state.status == ProxyStatus.STARTING
    proxy_state._actor_proxy_wrapper.ready = ProxyWrapperCallStatus.PENDING
    wait_for_condition(condition_predictor=_update_and_check_proxy_status, state=proxy_state, status=ProxyStatus.STARTING)
    proxy_state._actor_proxy_wrapper.ready = ProxyWrapperCallStatus.FINISHED_SUCCEED
    wait_for_condition(condition_predictor=_update_and_check_proxy_status, state=proxy_state, status=ProxyStatus.HEALTHY)

def test_proxy_state_update_starting_ready_always_fails():
    if False:
        print('Hello World!')
    'Test calling update method on ProxyState when the proxy state is STARTING and\n    when the ready call is always failing.\n\n    The proxy state started with STARTING. After update is called, read call only throws\n    exceptions. The state will eventually change to UNHEALTHY after all retries have\n    exhausted.\n    '
    proxy_state = _create_proxy_state()
    proxy_state._actor_proxy_wrapper.ready = ProxyWrapperCallStatus.FINISHED_FAILED
    assert proxy_state.status == ProxyStatus.STARTING
    wait_for_condition(condition_predictor=_update_and_check_proxy_status, state=proxy_state, status=ProxyStatus.UNHEALTHY)

@patch('ray.serve._private.proxy_state.PROXY_READY_CHECK_TIMEOUT_S', 0)
def test_proxy_state_update_starting_ready_always_timeout():
    if False:
        return 10
    'Test calling update method on ProxyState when the proxy state is STARTING and\n    when the ready call always timed out.\n\n    The proxy state started with STARTING. After update is called, ready calls takes\n    very long time to finish. The state will eventually change to UNHEALTHY after all\n    retries have exhausted.\n    '
    proxy_state = _create_proxy_state()
    proxy_state._actor_proxy_wrapper.ready = ProxyWrapperCallStatus.PENDING
    assert proxy_state.status == ProxyStatus.STARTING
    wait_for_condition(condition_predictor=_update_and_check_proxy_status, state=proxy_state, status=ProxyStatus.UNHEALTHY)

@patch('ray.serve._private.proxy_state.PROXY_HEALTH_CHECK_PERIOD_S', 0)
def test_proxy_state_update_healthy_check_health_succeed():
    if False:
        return 10
    'Test calling update method on ProxyState when the proxy state is HEALTHY and\n    when the check_health call succeeded\n\n    The proxy state started with HEALTHY. After update is called and ready call\n    succeeded, the status will change to HEALTHY. After the next period of check_health\n    call, the status should stay as HEALTHY.\n    '
    proxy_state = _create_proxy_state()
    wait_for_condition(condition_predictor=_update_and_check_proxy_status, state=proxy_state, status=ProxyStatus.HEALTHY)
    first_check_time = proxy_state._last_health_check_time
    for _ in range(3):
        _update_and_check_proxy_status(proxy_state, ProxyStatus.HEALTHY)
        time.sleep(0.1)
    assert first_check_time != proxy_state._last_health_check_time

@patch('ray.serve._private.proxy_state.PROXY_HEALTH_CHECK_PERIOD_S', 0)
def test_proxy_state_update_healthy_check_health_failed_once():
    if False:
        return 10
    'Test calling update method on ProxyState when the proxy state is HEALTHY and\n    when the check_health call failed once and succeeded for the following call.\n\n    The proxy state started with STARTING. After update is called and ready call\n    succeeded, the status will change to HEALTHY. After the next period of check_health\n    call and that check_health call failed, the status should not be set to UNHEALTHY\n    and should stay as HEALTHY. The following check_health call continue to succeed\n    and the status continue to stay as HEALTHY.\n    '
    proxy_state = _create_proxy_state()
    proxy_state._actor_proxy_wrapper.health = ProxyWrapperCallStatus.FINISHED_FAILED
    wait_for_condition(condition_predictor=_update_and_check_proxy_status, state=proxy_state, status=ProxyStatus.HEALTHY)
    _update_and_check_proxy_status(proxy_state, ProxyStatus.HEALTHY)
    proxy_state._actor_proxy_wrapper.health = ProxyWrapperCallStatus.FINISHED_SUCCEED
    wait_for_condition(condition_predictor=_update_and_check_proxy_status, state=proxy_state, status=ProxyStatus.HEALTHY)

@patch('ray.serve._private.proxy_state.PROXY_HEALTH_CHECK_PERIOD_S', 0)
def test_proxy_state_update_healthy_check_health_always_fails():
    if False:
        while True:
            i = 10
    'Test calling update method on ProxyState when the proxy state is HEALTHY and\n    when the check_health call is always failing.\n\n    The proxy state started with STARTING. After update is called and ready call\n    succeeded, the status will change to HEALTHY. After the next few check_health called\n    and failed, the status will eventually change to UNHEALTHY after all retries have\n    exhausted.\n    '
    proxy_state = _create_proxy_state()
    proxy_state._actor_proxy_wrapper.health = ProxyWrapperCallStatus.FINISHED_FAILED
    wait_for_condition(condition_predictor=_update_and_check_proxy_status, state=proxy_state, status=ProxyStatus.HEALTHY)
    first_check_time = proxy_state._last_health_check_time
    wait_for_condition(condition_predictor=_update_and_check_proxy_status, state=proxy_state, status=ProxyStatus.UNHEALTHY)
    assert first_check_time != proxy_state._last_health_check_time
    assert proxy_state._consecutive_health_check_failures == 3

@patch('ray.serve._private.proxy_state.PROXY_HEALTH_CHECK_PERIOD_S', 0)
def test_proxy_state_update_healthy_check_health_sometimes_fails():
    if False:
        return 10
    "Test that the proxy is UNHEALTHY after consecutive health-check failures.\n\n    The proxy state starts with STARTING. Then the proxy fails a few times\n    (less than the threshold needed to set it UNHEALTHY). Then it succeeds, so\n    it becomes HEALTHY. Then it fails a few times again but stays HEALTHY\n    because the failures weren't consecutive with the previous ones. And then\n    it finally fails enough times to become UNHEALTHY.\n    "
    proxy_state = _create_proxy_state()
    wait_for_condition(condition_predictor=_update_and_check_proxy_status, state=proxy_state, status=ProxyStatus.HEALTHY)

    def _update_until_num_health_checks_received(state: ProxyState, num_health_checks: int):
        if False:
            for i in range(10):
                print('nop')
        state.update()
        assert state._actor_proxy_wrapper.get_num_health_checks() == num_health_checks
        return True

    def incur_health_checks(pass_checks: bool, num_checks: int, expected_final_status: ProxyStatus):
        if False:
            print('Hello World!')
        'Waits for num_checks health checks to occur.\n\n        Args:\n            pass_checks: whether the health checks should pass.\n            num_checks: number of checks to wait for.\n            expected_final_status: the final status that should be asserted.\n        '
        if pass_checks:
            proxy_state._actor_proxy_wrapper.health = ProxyWrapperCallStatus.FINISHED_SUCCEED
        else:
            proxy_state._actor_proxy_wrapper.health = ProxyWrapperCallStatus.FINISHED_FAILED
        cur_num_health_checks = proxy_state._actor_proxy_wrapper.get_num_health_checks()
        wait_for_condition(condition_predictor=_update_until_num_health_checks_received, state=proxy_state, num_health_checks=cur_num_health_checks + num_checks)
        assert proxy_state._actor_proxy_wrapper.get_num_health_checks() <= cur_num_health_checks + num_checks
        if expected_final_status:
            assert proxy_state.status == expected_final_status
    for _ in range(3):
        incur_health_checks(pass_checks=True, num_checks=1, expected_final_status=ProxyStatus.HEALTHY)
        incur_health_checks(pass_checks=False, num_checks=PROXY_HEALTH_CHECK_UNHEALTHY_THRESHOLD - 1, expected_final_status=ProxyStatus.HEALTHY)
    incur_health_checks(pass_checks=True, num_checks=1, expected_final_status=ProxyStatus.HEALTHY)
    incur_health_checks(pass_checks=False, num_checks=PROXY_HEALTH_CHECK_UNHEALTHY_THRESHOLD, expected_final_status=ProxyStatus.UNHEALTHY)

@patch('ray.serve._private.proxy_state.PROXY_HEALTH_CHECK_TIMEOUT_S', 0)
@patch('ray.serve._private.proxy_state.PROXY_HEALTH_CHECK_PERIOD_S', 0)
def test_proxy_state_check_health_always_timeout():
    if False:
        print('Hello World!')
    'Test calling update method on ProxyState when the proxy state is HEALTHY and\n    when the ready call always timed out and health check timeout and period equals.\n\n    The proxy state started with STARTING. After update is called and ready call\n    succeeded, the status will change to HEALTHY. After the next few check_health calls\n    never finishes and always pending, the status will eventually change to UNHEALTHY\n    after all retries have exhausted.\n    '
    proxy_state = _create_proxy_state()
    proxy_state._actor_proxy_wrapper.health = ProxyWrapperCallStatus.PENDING
    wait_for_condition(condition_predictor=_update_and_check_proxy_status, state=proxy_state, status=ProxyStatus.HEALTHY)
    first_check_time = proxy_state._last_health_check_time
    wait_for_condition(condition_predictor=_update_and_check_proxy_status, state=proxy_state, status=ProxyStatus.UNHEALTHY)
    assert first_check_time != proxy_state._last_health_check_time
    assert proxy_state._consecutive_health_check_failures == 3

@patch('ray.serve._private.proxy_state.PROXY_HEALTH_CHECK_PERIOD_S', 0)
def test_proxy_state_update_unhealthy_check_health_succeed():
    if False:
        i = 10
        return i + 15
    'Test calling update method on ProxyState when the proxy state has\n    failed health checks and the next check_health call succeeded.\n\n    The proxy state started with STARTING. After the next period of check_health\n    call, the status changes to HEALTHY.\n    '
    proxy_state = _create_proxy_state()
    proxy_state._consecutive_health_check_failures = 1
    assert proxy_state.status == ProxyStatus.STARTING
    wait_for_condition(condition_predictor=_update_and_check_proxy_status, state=proxy_state, status=ProxyStatus.HEALTHY)
    assert proxy_state._consecutive_health_check_failures == 0

@patch('ray.serve._private.proxy_state.PROXY_HEALTH_CHECK_TIMEOUT_S', 0)
@patch('ray.serve._private.proxy_state.PROXY_HEALTH_CHECK_PERIOD_S', 0)
def test_unhealthy_retry_correct_number_of_times():
    if False:
        print('Hello World!')
    'Test the unhealthy retry logic retires the correct number of times.\n\n    When the health check fails 3 times (default retry threshold), the proxy state\n    should change from HEALTHY to UNHEALTHY.\n    '
    proxy_state = _create_proxy_state()
    proxy_state._actor_proxy_wrapper.health = ProxyWrapperCallStatus.PENDING
    proxy_state.update()
    assert proxy_state.status == ProxyStatus.HEALTHY

    def proxy_state_consecutive_health_check_failures(num_failures):
        if False:
            i = 10
            return i + 15
        proxy_state.update()
        assert proxy_state._consecutive_health_check_failures == num_failures
        return True
    wait_for_condition(condition_predictor=proxy_state_consecutive_health_check_failures, num_failures=3)
    assert proxy_state.status == ProxyStatus.UNHEALTHY

@patch('ray.serve._private.proxy_state.PROXY_HEALTH_CHECK_PERIOD_S', 0)
@pytest.mark.parametrize('number_of_worker_nodes', [0, 1, 2, 3])
def test_update_draining(all_nodes, number_of_worker_nodes):
    if False:
        print('Hello World!')
    'Test update draining logics.\n\n    When update nodes to inactive, head node http proxy should never be draining while\n    worker node http proxy should change to draining. When update nodes to active, head\n    node http proxy should continue to be healthy while worker node http proxy should\n    be healthy.\n    '
    (manager, cluster_node_info_cache) = _create_proxy_state_manager(HTTPOptions(location=DeploymentMode.EveryNode))
    cluster_node_info_cache.alive_nodes = all_nodes
    for (node_id, _) in all_nodes:
        manager._proxy_states[node_id] = _create_proxy_state(status=ProxyStatus.HEALTHY, node_id=node_id)
    node_ids = [node_id for (node_id, _) in all_nodes]
    proxy_nodes = set()
    wait_for_condition(condition_predictor=_update_and_check_proxy_state_manager, proxy_state_manager=manager, node_ids=node_ids, statuses=[ProxyStatus.HEALTHY] + [ProxyStatus.DRAINING] * number_of_worker_nodes, proxy_nodes=proxy_nodes)
    proxy_nodes = set(node_ids)
    wait_for_condition(condition_predictor=_update_and_check_proxy_state_manager, proxy_state_manager=manager, node_ids=node_ids, statuses=[ProxyStatus.HEALTHY] * (number_of_worker_nodes + 1), proxy_nodes=proxy_nodes)

@patch('ray.serve._private.proxy_state.PROXY_HEALTH_CHECK_PERIOD_S', 0)
def test_proxy_actor_healthy_during_draining():
    if False:
        while True:
            i = 10
    'Test that the proxy will remain DRAINING even if health check succeeds.'
    proxy_state = _create_proxy_state()
    wait_for_condition(condition_predictor=_update_and_check_proxy_status, state=proxy_state, status=ProxyStatus.HEALTHY)
    proxy_state.update(draining=True)
    assert proxy_state.status == ProxyStatus.DRAINING
    cur_num_health_checks = proxy_state._actor_proxy_wrapper.get_num_health_checks()

    def _update_until_two_more_health_checks():
        if False:
            for i in range(10):
                print('nop')
        proxy_state.update(draining=True)
        return proxy_state._actor_proxy_wrapper.get_num_health_checks() == cur_num_health_checks + 2
    wait_for_condition(_update_until_two_more_health_checks)
    assert proxy_state.status == ProxyStatus.DRAINING

@patch('ray.serve._private.proxy_state.PROXY_HEALTH_CHECK_PERIOD_S', 0)
@patch('ray.serve._private.proxy_state.PROXY_DRAIN_CHECK_PERIOD_S', 0)
@pytest.mark.parametrize('number_of_worker_nodes', [1])
def test_proxy_actor_unhealthy_during_draining(all_nodes, number_of_worker_nodes):
    if False:
        print('Hello World!')
    'Test the state transition from DRAINING to UNHEALTHY for the proxy actor.'
    (manager, cluster_node_info_cache) = _create_proxy_state_manager(HTTPOptions(location=DeploymentMode.EveryNode))
    cluster_node_info_cache.alive_nodes = all_nodes
    worker_node_id = None
    for (node_id, node_ip_address) in all_nodes:
        manager._proxy_states[node_id] = _create_proxy_state(status=ProxyStatus.STARTING, node_id=node_id)
        if node_id != HEAD_NODE_ID:
            worker_node_id = node_id
    node_ids = [node_id for (node_id, _) in all_nodes]
    proxy_nodes = set(node_ids)
    wait_for_condition(condition_predictor=_update_and_check_proxy_state_manager, proxy_state_manager=manager, node_ids=node_ids, statuses=[ProxyStatus.HEALTHY] * (number_of_worker_nodes + 1), proxy_nodes=proxy_nodes)
    proxy_nodes = set()
    wait_for_condition(condition_predictor=_update_and_check_proxy_state_manager, proxy_state_manager=manager, node_ids=node_ids, statuses=[ProxyStatus.HEALTHY] + [ProxyStatus.DRAINING] * number_of_worker_nodes, proxy_nodes=proxy_nodes)
    manager._proxy_states[worker_node_id]._actor_proxy_wrapper.health = ProxyWrapperCallStatus.FINISHED_FAILED

    def check_worker_node_proxy_actor_is_removed():
        if False:
            for i in range(10):
                print('nop')
        manager.update(proxy_nodes=proxy_nodes)
        return len(manager._proxy_states) == 1
    wait_for_condition(condition_predictor=check_worker_node_proxy_actor_is_removed)
    assert manager._proxy_states[HEAD_NODE_ID].status == ProxyStatus.HEALTHY

def test_is_ready_for_shutdown(all_nodes):
    if False:
        for i in range(10):
            print('nop')
    'Test `is_ready_for_shutdown()` returns True the correct state.\n\n    Before `shutdown()` is called, `is_ready_for_shutdown()` should return false. After\n    `shutdown()` is called and all proxy actor are killed, `is_ready_for_shutdown()`\n    should return true.\n    '
    (manager, cluster_node_info_cache) = _create_proxy_state_manager(HTTPOptions(location=DeploymentMode.EveryNode))
    cluster_node_info_cache.alive_nodes = all_nodes
    for (node_id, node_ip_address) in all_nodes:
        manager._proxy_states[node_id] = _create_proxy_state(status=ProxyStatus.HEALTHY, node_id=node_id)
    assert not manager.is_ready_for_shutdown()
    manager.shutdown()

    def check_is_ready_for_shutdown():
        if False:
            return 10
        return manager.is_ready_for_shutdown()
    wait_for_condition(check_is_ready_for_shutdown)

@patch('ray.serve._private.proxy_state.PROXY_READY_CHECK_TIMEOUT_S', 0.1)
@pytest.mark.parametrize('number_of_worker_nodes', [1])
def test_proxy_starting_timeout_longer_than_env(number_of_worker_nodes, all_nodes):
    if False:
        return 10
    'Test update method on ProxyStateManager when the proxy state is STARTING and\n    when the ready call takes longer than PROXY_READY_CHECK_TIMEOUT_S.\n\n    The proxy state started with STARTING. After update is called, ready calls takes\n    some time to finish. The proxy state manager will restart the proxy state after\n    PROXY_READY_CHECK_TIMEOUT_S. After the next period of check_health call,\n    the proxy state manager will check on backoff timeout, not immediately\n    restarting the proxy states, and eventually set the proxy state to HEALTHY.\n    '
    fake_time = MockTimer()
    (proxy_state_manager, cluster_node_info_cache) = _create_proxy_state_manager(http_options=HTTPOptions(location=DeploymentMode.EveryNode), timer=fake_time)
    cluster_node_info_cache.alive_nodes = all_nodes
    node_ids = {node[0] for node in all_nodes}
    proxy_state_manager.update(proxy_nodes=node_ids)
    old_proxy_states = {node_id: state for (node_id, state) in proxy_state_manager._proxy_states.items()}
    assert len(proxy_state_manager._proxy_states) == len(node_ids)

    def check_proxy_state_starting(_proxy_state_manager: ProxyStateManager):
        if False:
            while True:
                i = 10
        for proxy_state in _proxy_state_manager._proxy_states.values():
            assert proxy_state.status == ProxyStatus.STARTING
            proxy_state._actor_proxy_wrapper.ready = ProxyWrapperCallStatus.PENDING
    check_proxy_state_starting(_proxy_state_manager=proxy_state_manager)
    fake_time.advance(0.11)
    proxy_state_manager.update(proxy_nodes=node_ids)
    assert all([proxy_state_manager._proxy_states[node_id] != old_proxy_states[node_id] for node_id in node_ids])
    old_proxy_states = {node_id: state for (node_id, state) in proxy_state_manager._proxy_states.items()}
    check_proxy_state_starting(_proxy_state_manager=proxy_state_manager)
    fake_time.advance(0.11)
    proxy_state_manager.update(proxy_nodes=node_ids)
    assert all([proxy_state_manager._proxy_states[node_id] == old_proxy_states[node_id] for node_id in node_ids])
    for proxy_state in proxy_state_manager._proxy_states.values():
        proxy_state._actor_proxy_wrapper.ready = ProxyWrapperCallStatus.FINISHED_SUCCEED
    proxy_state_manager.update(proxy_nodes=node_ids)
    assert all([proxy_state_manager._proxy_states[node_id].status == ProxyStatus.HEALTHY for node_id in node_ids])
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', '-s', __file__]))