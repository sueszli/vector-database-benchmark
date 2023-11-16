import logging
import os
import sys
import tempfile
import time
import zipfile
from typing import Iterable, List
from unittest import mock
import numpy as np
import pytest
import requests
import ray
import ray.util.state as state_api
from ray import serve
from ray._private.test_utils import SignalActor, wait_for_condition
from ray.serve._private.autoscaling_policy import BasicAutoscalingPolicy, calculate_desired_num_replicas
from ray.serve._private.common import DeploymentID, DeploymentStatus, DeploymentStatusInfo, ReplicaState
from ray.serve._private.constants import CONTROL_LOOP_PERIOD_S, SERVE_DEFAULT_APP_NAME
from ray.serve._private.controller import ServeController
from ray.serve.config import AutoscalingConfig
from ray.serve.generated.serve_pb2 import DeploymentStatusInfo as DeploymentStatusInfoProto
from ray.serve.schema import ServeDeploySchema

class TestCalculateDesiredNumReplicas:

    def test_bounds_checking(self):
        if False:
            print('Hello World!')
        num_replicas = 10
        max_replicas = 11
        min_replicas = 9
        config = AutoscalingConfig(max_replicas=max_replicas, min_replicas=min_replicas, target_num_ongoing_requests_per_replica=100)
        desired_num_replicas = calculate_desired_num_replicas(autoscaling_config=config, current_num_ongoing_requests=[150] * num_replicas)
        assert desired_num_replicas == max_replicas
        desired_num_replicas = calculate_desired_num_replicas(autoscaling_config=config, current_num_ongoing_requests=[50] * num_replicas)
        assert desired_num_replicas == min_replicas
        for i in range(50, 150):
            desired_num_replicas = calculate_desired_num_replicas(autoscaling_config=config, current_num_ongoing_requests=[i] * num_replicas)
            assert min_replicas <= desired_num_replicas <= max_replicas

    @pytest.mark.parametrize('target_requests', [0.5, 1.0, 1.5])
    def test_scale_up(self, target_requests):
        if False:
            print('Hello World!')
        config = AutoscalingConfig(min_replicas=0, max_replicas=100, target_num_ongoing_requests_per_replica=target_requests)
        num_replicas = 10
        num_ongoing_requests = [2 * target_requests] * num_replicas
        desired_num_replicas = calculate_desired_num_replicas(autoscaling_config=config, current_num_ongoing_requests=num_ongoing_requests)
        assert 19 <= desired_num_replicas <= 21

    @pytest.mark.parametrize('target_requests', [0.5, 1.0, 1.5])
    def test_scale_down(self, target_requests):
        if False:
            while True:
                i = 10
        config = AutoscalingConfig(min_replicas=0, max_replicas=100, target_num_ongoing_requests_per_replica=target_requests)
        num_replicas = 10
        num_ongoing_requests = [0.5 * target_requests] * num_replicas
        desired_num_replicas = calculate_desired_num_replicas(autoscaling_config=config, current_num_ongoing_requests=num_ongoing_requests)
        assert 4 <= desired_num_replicas <= 6

    def test_smoothing_factor(self):
        if False:
            i = 10
            return i + 15
        config = AutoscalingConfig(min_replicas=0, max_replicas=100, target_num_ongoing_requests_per_replica=1, smoothing_factor=0.5)
        num_replicas = 10
        num_ongoing_requests = [4.0] * num_replicas
        desired_num_replicas = calculate_desired_num_replicas(autoscaling_config=config, current_num_ongoing_requests=num_ongoing_requests)
        assert 24 <= desired_num_replicas <= 26
        num_ongoing_requests = [0.25] * num_replicas
        desired_num_replicas = calculate_desired_num_replicas(autoscaling_config=config, current_num_ongoing_requests=num_ongoing_requests)
        assert 5 <= desired_num_replicas <= 8

    def test_upscale_smoothing_factor(self):
        if False:
            while True:
                i = 10
        config = AutoscalingConfig(min_replicas=0, max_replicas=100, target_num_ongoing_requests_per_replica=1, upscale_smoothing_factor=0.5)
        num_replicas = 10
        num_ongoing_requests = [4.0] * num_replicas
        desired_num_replicas = calculate_desired_num_replicas(autoscaling_config=config, current_num_ongoing_requests=num_ongoing_requests)
        assert 24 <= desired_num_replicas <= 26
        num_ongoing_requests = [0.25] * num_replicas
        desired_num_replicas = calculate_desired_num_replicas(autoscaling_config=config, current_num_ongoing_requests=num_ongoing_requests)
        assert 1 <= desired_num_replicas <= 4

    def test_downscale_smoothing_factor(self):
        if False:
            i = 10
            return i + 15
        config = AutoscalingConfig(min_replicas=0, max_replicas=100, target_num_ongoing_requests_per_replica=1, downscale_smoothing_factor=0.5)
        num_replicas = 10
        num_ongoing_requests = [4.0] * num_replicas
        desired_num_replicas = calculate_desired_num_replicas(autoscaling_config=config, current_num_ongoing_requests=num_ongoing_requests)
        assert 39 <= desired_num_replicas <= 41
        num_ongoing_requests = [0.25] * num_replicas
        desired_num_replicas = calculate_desired_num_replicas(autoscaling_config=config, current_num_ongoing_requests=num_ongoing_requests)
        assert 5 <= desired_num_replicas <= 8

class TestGetDecisionNumReplicas:

    def test_smoothing_factor_scale_up_from_0_replicas(self):
        if False:
            return 10
        'Test that the smoothing factor is respected when scaling up\n        from 0 replicas.\n        '
        config = AutoscalingConfig(min_replicas=0, max_replicas=2, smoothing_factor=10)
        policy = BasicAutoscalingPolicy(config)
        new_num_replicas = policy.get_decision_num_replicas(current_num_ongoing_requests=[], curr_target_num_replicas=0, current_handle_queued_queries=1)
        assert new_num_replicas == 10
        config.smoothing_factor = 0.5
        policy = BasicAutoscalingPolicy(config)
        new_num_replicas = policy.get_decision_num_replicas(current_num_ongoing_requests=[], curr_target_num_replicas=0, current_handle_queued_queries=1)
        assert new_num_replicas == 1

    def test_smoothing_factor_scale_down_to_0_replicas(self):
        if False:
            while True:
                i = 10
        'Test that a deployment scales down to 0 for non-default smoothing factors.'
        config = AutoscalingConfig(min_replicas=0, max_replicas=5, smoothing_factor=10, upscale_delay_s=0, downscale_delay_s=0)
        policy = BasicAutoscalingPolicy(config)
        new_num_replicas = policy.get_decision_num_replicas(current_num_ongoing_requests=[0, 0, 0, 0, 0], curr_target_num_replicas=5, current_handle_queued_queries=0)
        assert new_num_replicas == 0
        config.smoothing_factor = 0.2
        policy = BasicAutoscalingPolicy(config)
        num_replicas = 5
        for _ in range(5):
            num_replicas = policy.get_decision_num_replicas(current_num_ongoing_requests=[0] * num_replicas, curr_target_num_replicas=num_replicas, current_handle_queued_queries=0)
        assert num_replicas == 0

    def test_upscale_downscale_delay(self):
        if False:
            for i in range(10):
                print('nop')
        'Unit test for upscale_delay_s and downscale_delay_s.'
        upscale_delay_s = 30.0
        downscale_delay_s = 600.0
        config = AutoscalingConfig(min_replicas=0, max_replicas=2, target_num_ongoing_requests_per_replica=1, upscale_delay_s=30.0, downscale_delay_s=600.0)
        policy = BasicAutoscalingPolicy(config)
        upscale_wait_periods = int(upscale_delay_s / CONTROL_LOOP_PERIOD_S)
        downscale_wait_periods = int(downscale_delay_s / CONTROL_LOOP_PERIOD_S)
        overload_requests = [100]
        new_num_replicas = policy.get_decision_num_replicas(current_num_ongoing_requests=[], curr_target_num_replicas=0, current_handle_queued_queries=1)
        assert new_num_replicas == 1
        for i in range(upscale_wait_periods):
            new_num_replicas = policy.get_decision_num_replicas(current_num_ongoing_requests=overload_requests, curr_target_num_replicas=1, current_handle_queued_queries=0)
            assert new_num_replicas == 1, i
        new_num_replicas = policy.get_decision_num_replicas(current_num_ongoing_requests=overload_requests, curr_target_num_replicas=1, current_handle_queued_queries=0)
        assert new_num_replicas == 2
        no_requests = [0, 0]
        for i in range(downscale_wait_periods):
            new_num_replicas = policy.get_decision_num_replicas(current_num_ongoing_requests=no_requests, curr_target_num_replicas=2, current_handle_queued_queries=0)
            assert new_num_replicas == 2, i
        new_num_replicas = policy.get_decision_num_replicas(current_num_ongoing_requests=no_requests, curr_target_num_replicas=2, current_handle_queued_queries=0)
        assert new_num_replicas == 0
        for i in range(int(upscale_wait_periods / 2)):
            new_num_replicas = policy.get_decision_num_replicas(current_num_ongoing_requests=overload_requests, curr_target_num_replicas=1, current_handle_queued_queries=0)
            assert new_num_replicas == 1, i
        policy.get_decision_num_replicas(current_num_ongoing_requests=[0], curr_target_num_replicas=1, current_handle_queued_queries=0)
        for i in range(upscale_wait_periods):
            new_num_replicas = policy.get_decision_num_replicas(current_num_ongoing_requests=overload_requests, curr_target_num_replicas=1, current_handle_queued_queries=0)
            assert new_num_replicas == 1, i
        new_num_replicas = policy.get_decision_num_replicas(current_num_ongoing_requests=overload_requests, curr_target_num_replicas=1, current_handle_queued_queries=0)
        assert new_num_replicas == 2
        for i in range(int(downscale_wait_periods / 2)):
            new_num_replicas = policy.get_decision_num_replicas(current_num_ongoing_requests=no_requests, curr_target_num_replicas=2, current_handle_queued_queries=0)
            assert new_num_replicas == 2, i
        policy.get_decision_num_replicas(current_num_ongoing_requests=[100, 100], curr_target_num_replicas=2, current_handle_queued_queries=0)
        for i in range(downscale_wait_periods):
            new_num_replicas = policy.get_decision_num_replicas(current_num_ongoing_requests=no_requests, curr_target_num_replicas=2, current_handle_queued_queries=0)
            assert new_num_replicas == 2, i
        new_num_replicas = policy.get_decision_num_replicas(current_num_ongoing_requests=no_requests, curr_target_num_replicas=2, current_handle_queued_queries=0)
        assert new_num_replicas == 0

    def test_replicas_delayed_startup(self):
        if False:
            print('Hello World!')
        'Unit test simulating replicas taking time to start up.'
        config = AutoscalingConfig(min_replicas=1, max_replicas=200, target_num_ongoing_requests_per_replica=1, upscale_delay_s=0, downscale_delay_s=100000)
        policy = BasicAutoscalingPolicy(config)
        new_num_replicas = policy.get_decision_num_replicas(1, [100], 0)
        assert new_num_replicas == 100
        new_num_replicas = policy.get_decision_num_replicas(100, [100], 0)
        assert new_num_replicas == 100
        new_num_replicas = policy.get_decision_num_replicas(100, [100, 20, 3], 0)
        assert new_num_replicas == 123
        new_num_replicas = policy.get_decision_num_replicas(123, [6, 2, 1, 1], 0)
        assert new_num_replicas == 123

    @pytest.mark.parametrize('delay_s', [30.0, 0.0])
    def test_fluctuating_ongoing_requests(self, delay_s):
        if False:
            for i in range(10):
                print('nop')
        '\n        Simulates a workload that switches between too many and too few\n        ongoing requests.\n        '
        config = AutoscalingConfig(min_replicas=1, max_replicas=10, target_num_ongoing_requests_per_replica=50, upscale_delay_s=delay_s, downscale_delay_s=delay_s)
        policy = BasicAutoscalingPolicy(config)
        if delay_s > 0:
            wait_periods = int(delay_s / CONTROL_LOOP_PERIOD_S)
            assert wait_periods > 1
        (underload_requests, overload_requests) = ([20, 20], [100])
        trials = 1000
        new_num_replicas = None
        for trial in range(trials):
            if trial % 2 == 0:
                new_num_replicas = policy.get_decision_num_replicas(current_num_ongoing_requests=overload_requests, curr_target_num_replicas=1, current_handle_queued_queries=0)
                if delay_s > 0:
                    assert new_num_replicas == 1, trial
                else:
                    assert new_num_replicas == 2, trial
            else:
                new_num_replicas = policy.get_decision_num_replicas(current_num_ongoing_requests=underload_requests, curr_target_num_replicas=2, current_handle_queued_queries=0)
                if delay_s > 0:
                    assert new_num_replicas == 2, trial
                else:
                    assert new_num_replicas == 1, trial

    @pytest.mark.parametrize('ongoing_requests', [[7, 1, 8, 4], [8, 1, 8, 4], [6, 1, 8, 4], [0, 1, 8, 4]])
    def test_imbalanced_replicas(self, ongoing_requests):
        if False:
            for i in range(10):
                print('nop')
        config = AutoscalingConfig(min_replicas=1, max_replicas=10, target_num_ongoing_requests_per_replica=5, upscale_delay_s=0.0, downscale_delay_s=0.0)
        policy = BasicAutoscalingPolicy(config)
        if np.mean(ongoing_requests) == config.target_num_ongoing_requests_per_replica:
            new_num_replicas = policy.get_decision_num_replicas(current_num_ongoing_requests=ongoing_requests, curr_target_num_replicas=4, current_handle_queued_queries=0)
            assert new_num_replicas == 4
        elif np.mean(ongoing_requests) < config.target_num_ongoing_requests_per_replica:
            new_num_replicas = policy.get_decision_num_replicas(current_num_ongoing_requests=ongoing_requests, curr_target_num_replicas=4, current_handle_queued_queries=0)
            if config.target_num_ongoing_requests_per_replica - np.mean(ongoing_requests) <= 1:
                assert new_num_replicas == 4
            else:
                assert new_num_replicas == 3
        else:
            new_num_replicas = policy.get_decision_num_replicas(current_num_ongoing_requests=ongoing_requests, curr_target_num_replicas=4, current_handle_queued_queries=0)
            assert new_num_replicas == 5

    @pytest.mark.parametrize('ongoing_requests', [[20, 0, 0, 0], [100, 0, 0, 0], [10, 0, 0, 0]])
    def test_single_replica_receives_all_requests(self, ongoing_requests):
        if False:
            print('Hello World!')
        target_requests = 5
        config = AutoscalingConfig(min_replicas=1, max_replicas=50, target_num_ongoing_requests_per_replica=target_requests, upscale_delay_s=0.0, downscale_delay_s=0.0)
        policy = BasicAutoscalingPolicy(config)
        new_num_replicas = policy.get_decision_num_replicas(current_num_ongoing_requests=ongoing_requests, curr_target_num_replicas=4, current_handle_queued_queries=0)
        assert new_num_replicas == sum(ongoing_requests) / target_requests

def get_deployment_status(controller, name) -> DeploymentStatus:
    if False:
        return 10
    ref = ray.get(controller.get_deployment_status.remote(name, SERVE_DEFAULT_APP_NAME))
    info = DeploymentStatusInfo.from_proto(DeploymentStatusInfoProto.FromString(ref))
    return info.status

def get_running_replicas(controller: ServeController, name: str) -> List:
    if False:
        while True:
            i = 10
    'Get the replicas currently running for given deployment'
    replicas = ray.get(controller._dump_replica_states_for_testing.remote(DeploymentID(name, SERVE_DEFAULT_APP_NAME)))
    running_replicas = replicas.get([ReplicaState.RUNNING])
    return running_replicas

def get_running_replica_tags(controller: ServeController, name: str) -> List:
    if False:
        while True:
            i = 10
    'Get the replica tags of running replicas for given deployment'
    running_replicas = get_running_replicas(controller, name)
    return [replica.replica_tag for replica in running_replicas]

def check_autoscale_num_replicas(controller: ServeController, name: str) -> int:
    if False:
        return 10
    'Check the number of replicas currently running for given deployment.\n\n    This should only be called if the deployment has already transitioned\n    to HEALTHY, and this function will check that it remains healthy.\n    '
    assert get_deployment_status(controller, name) == DeploymentStatus.HEALTHY
    return len(get_running_replicas(controller, name))

def assert_no_replicas_deprovisioned(replica_tags_1: Iterable[str], replica_tags_2: Iterable[str]) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Checks whether any replica tags from replica_tags_1 are absent from\n    replica_tags_2. Assumes that this indicates replicas were de-provisioned.\n\n    replica_tags_1: Replica tags of running replicas at the first timestep\n    replica_tags_2: Replica tags of running replicas at the second timestep\n    '
    (replica_tags_1, replica_tags_2) = (set(replica_tags_1), set(replica_tags_2))
    num_matching_replicas = len(replica_tags_1.intersection(replica_tags_2))
    print(f'{num_matching_replicas} replica(s) stayed provisioned between both deployments. All {len(replica_tags_1)} replica(s) were expected to stay provisioned. {len(replica_tags_1) - num_matching_replicas} replica(s) were de-provisioned.')
    assert len(replica_tags_1) == num_matching_replicas

def test_assert_no_replicas_deprovisioned():
    if False:
        for i in range(10):
            print('nop')
    replica_tags_1 = ['a', 'b', 'c']
    replica_tags_2 = ['a', 'b', 'c', 'd', 'e']
    assert_no_replicas_deprovisioned(replica_tags_1, replica_tags_2)
    with pytest.raises(AssertionError):
        assert_no_replicas_deprovisioned(replica_tags_2, replica_tags_1)

def get_deployment_start_time(controller: ServeController, name: str):
    if False:
        for i in range(10):
            print('nop')
    'Return start time for given deployment'
    deployments = ray.get(controller.list_deployments_internal.remote())
    (deployment_info, _) = deployments[DeploymentID(name, SERVE_DEFAULT_APP_NAME)]
    return deployment_info.start_time_ms

@pytest.mark.parametrize('min_replicas', [1, 2])
def test_e2e_scale_up_down_basic(min_replicas, serve_instance):
    if False:
        print('Hello World!')
    'Send 100 requests and check that we autoscale up, and then back down.'
    controller = serve_instance._controller
    signal = SignalActor.remote()

    @serve.deployment(autoscaling_config={'metrics_interval_s': 0.1, 'min_replicas': min_replicas, 'max_replicas': 3, 'look_back_period_s': 0.2, 'downscale_delay_s': 0.5, 'upscale_delay_s': 0}, graceful_shutdown_timeout_s=1, max_concurrent_queries=1000, version='v1')
    class A:

        def __call__(self):
            if False:
                return 10
            ray.get(signal.wait.remote())
    handle = serve.run(A.bind())
    wait_for_condition(lambda : get_deployment_status(controller, 'A') == DeploymentStatus.HEALTHY)
    start_time = get_deployment_start_time(controller, 'A')
    [handle.remote() for _ in range(100)]
    wait_for_condition(lambda : check_autoscale_num_replicas(controller, 'A') >= min_replicas + 1, raise_exceptions=True)
    signal.send.remote()
    wait_for_condition(lambda : check_autoscale_num_replicas(controller, 'A') <= min_replicas, raise_exceptions=True)
    assert get_deployment_start_time(controller, 'A') == start_time

@pytest.mark.skipif(sys.platform == 'win32', reason='Failing on Windows.')
@pytest.mark.parametrize('smoothing_factor', [1, 0.2])
@pytest.mark.parametrize('use_upscale_downscale_config', [True, False])
def test_e2e_scale_up_down_with_0_replica(serve_instance, smoothing_factor, use_upscale_downscale_config):
    if False:
        return 10
    'Send 100 requests and check that we autoscale up, and then back down.'
    controller = serve_instance._controller
    signal = SignalActor.remote()
    autoscaling_config = {'metrics_interval_s': 0.1, 'min_replicas': 0, 'max_replicas': 2, 'look_back_period_s': 0.2, 'downscale_delay_s': 0.5, 'upscale_delay_s': 0}
    if use_upscale_downscale_config:
        autoscaling_config['upscale_smoothing_factor'] = smoothing_factor
        autoscaling_config['downscale_smoothing_factor'] = smoothing_factor
    else:
        autoscaling_config['smoothing_factor'] = smoothing_factor

    @serve.deployment(autoscaling_config=autoscaling_config, graceful_shutdown_timeout_s=1, max_concurrent_queries=1000, version='v1')
    class A:

        def __call__(self):
            if False:
                while True:
                    i = 10
            ray.get(signal.wait.remote())
    handle = serve.run(A.bind()).options(use_new_handle_api=True)
    wait_for_condition(lambda : get_deployment_status(controller, 'A') == DeploymentStatus.HEALTHY)
    start_time = get_deployment_start_time(controller, 'A')
    results = [handle.remote() for _ in range(100)]
    wait_for_condition(lambda : check_autoscale_num_replicas(controller, 'A') >= 1, raise_exceptions=True)
    print('Number of replicas reached at least 1, releasing signal.')
    signal.send.remote()
    wait_for_condition(lambda : check_autoscale_num_replicas(controller, 'A') <= 0, raise_exceptions=True)
    for res in results:
        res.result()
    assert get_deployment_start_time(controller, 'A') == start_time

@mock.patch.object(ServeController, 'run_control_loop')
def test_initial_num_replicas(mock, serve_instance):
    if False:
        for i in range(10):
            print('nop')
    'assert that the inital amount of replicas a deployment is launched with\n    respects the bounds set by autoscaling_config.\n\n    For this test we mock out the run event loop, make sure the number of\n    replicas is set correctly before we hit the autoscaling procedure.\n    '

    @serve.deployment(autoscaling_config={'min_replicas': 2, 'max_replicas': 4}, version='v1')
    class A:

        def __call__(self):
            if False:
                print('Hello World!')
            return 'ok!'
    serve.run(A.bind())
    controller = serve_instance._controller
    assert len(get_running_replicas(controller, 'A')) == 2

def test_cold_start_time(serve_instance):
    if False:
        i = 10
        return i + 15
    "Test a request is served quickly by a deployment that's scaled to zero"

    @serve.deployment(autoscaling_config={'min_replicas': 0, 'max_replicas': 1, 'look_back_period_s': 0.2})
    class A:

        def __call__(self):
            if False:
                while True:
                    i = 10
            return 'hello'
    handle = serve.run(A.bind())

    def check_running():
        if False:
            for i in range(10):
                print('nop')
        assert serve.status().applications['default'].status == 'RUNNING'
        return True
    wait_for_condition(check_running)
    start = time.time()
    result = handle.remote().result()
    cold_start_time = time.time() - start
    assert cold_start_time < 3
    print('Time taken for deployment at 0 replicas to serve first request:', cold_start_time)
    assert result == 'hello'

@pytest.mark.skipif(sys.platform == 'win32', reason='Failing on Windows.')
def test_e2e_bursty(serve_instance):
    if False:
        return 10
    '\n    Sends 100 requests in bursts. Uses delays for smooth provisioning.\n    '
    controller = serve_instance._controller
    signal = SignalActor.remote()

    @serve.deployment(autoscaling_config={'metrics_interval_s': 0.1, 'min_replicas': 1, 'max_replicas': 2, 'look_back_period_s': 0.5, 'downscale_delay_s': 0.5, 'upscale_delay_s': 0.5}, graceful_shutdown_timeout_s=1, max_concurrent_queries=1000, version='v1')
    class A:

        def __init__(self):
            if False:
                print('Hello World!')
            logging.getLogger('ray.serve').setLevel(logging.ERROR)

        def __call__(self):
            if False:
                i = 10
                return i + 15
            ray.get(signal.wait.remote())
    handle = serve.run(A.bind())
    wait_for_condition(lambda : get_deployment_status(controller, 'A') == DeploymentStatus.HEALTHY)
    start_time = get_deployment_start_time(controller, 'A')
    [handle.remote() for _ in range(100)]
    wait_for_condition(lambda : check_autoscale_num_replicas(controller, 'A') >= 2, raise_exceptions=True)
    num_replicas = check_autoscale_num_replicas(controller, 'A')
    signal.send.remote()
    for _ in range(5):
        ray.get(signal.send.remote(clear=True))
        assert check_autoscale_num_replicas(controller, 'A') == num_replicas
        responses = [handle.remote() for _ in range(100)]
        signal.send.remote()
        [r.result() for r in responses]
        time.sleep(0.05)
    wait_for_condition(lambda : check_autoscale_num_replicas(controller, 'A') <= 1, raise_exceptions=True)
    assert get_deployment_start_time(controller, 'A') == start_time

@pytest.mark.skipif(sys.platform == 'win32', reason='Failing on Windows.')
def test_e2e_intermediate_downscaling(serve_instance):
    if False:
        while True:
            i = 10
    '\n    Scales up, then down, and up again.\n    '
    controller = serve_instance._controller
    signal = SignalActor.remote()

    @serve.deployment(autoscaling_config={'metrics_interval_s': 0.1, 'min_replicas': 0, 'max_replicas': 20, 'look_back_period_s': 0.2, 'downscale_delay_s': 0.2, 'upscale_delay_s': 0.2}, graceful_shutdown_timeout_s=1, max_concurrent_queries=1000, version='v1')
    class A:

        def __call__(self):
            if False:
                for i in range(10):
                    print('nop')
            ray.get(signal.wait.remote())
    handle = serve.run(A.bind())
    wait_for_condition(lambda : get_deployment_status(controller, 'A') == DeploymentStatus.HEALTHY)
    start_time = get_deployment_start_time(controller, 'A')
    [handle.remote() for _ in range(50)]
    wait_for_condition(lambda : check_autoscale_num_replicas(controller, 'A') >= 20, timeout=30, raise_exceptions=True)
    signal.send.remote()
    wait_for_condition(lambda : check_autoscale_num_replicas(controller, 'A') <= 1, timeout=30, raise_exceptions=True)
    signal.send.remote(clear=True)
    [handle.remote() for _ in range(50)]
    wait_for_condition(lambda : check_autoscale_num_replicas(controller, 'A') >= 20, timeout=30, raise_exceptions=True)
    signal.send.remote()
    wait_for_condition(lambda : check_autoscale_num_replicas(controller, 'A') < 1, timeout=30, raise_exceptions=True)
    assert get_deployment_start_time(controller, 'A') == start_time

@pytest.mark.skipif(sys.platform == 'win32', reason='Failing on Windows.')
@pytest.mark.skip(reason='Currently failing with undefined behavior')
def test_e2e_update_autoscaling_deployment(serve_instance):
    if False:
        i = 10
        return i + 15
    controller = serve_instance._controller
    signal = SignalActor.options(name='signal123').remote()
    app_config = {'import_path': 'ray.serve.tests.test_config_files.get_signal.app', 'deployments': [{'name': 'A', 'autoscaling_config': {'metrics_interval_s': 0.1, 'min_replicas': 0, 'max_replicas': 10, 'look_back_period_s': 0.2, 'downscale_delay_s': 0.2, 'upscale_delay_s': 0.2}, 'graceful_shutdown_timeout_s': 1, 'max_concurrent_queries': 1000}]}
    serve_instance.deploy_apps(ServeDeploySchema(**{'applications': [app_config]}))
    print('Deployed A with min_replicas 1 and max_replicas 10.')
    wait_for_condition(lambda : get_deployment_status(controller, 'A') == DeploymentStatus.HEALTHY)
    handle = serve.get_deployment_handle('A', 'default')
    start_time = get_deployment_start_time(controller, 'A')
    assert check_autoscale_num_replicas(controller, 'A') == 0
    [handle.remote() for _ in range(400)]
    print('Issued 400 requests.')
    wait_for_condition(lambda : check_autoscale_num_replicas(controller, 'A') >= 10, raise_exceptions=True)
    print('Scaled to 10 replicas.')
    first_deployment_replicas = get_running_replica_tags(controller, 'A')
    assert check_autoscale_num_replicas(controller, 'A') < 20
    [handle.remote() for _ in range(458)]
    time.sleep(3)
    print('Issued 458 requests. Request routing in-progress.')
    app_config['deployments'][0]['autoscaling_config']['min_replicas'] = 2
    app_config['deployments'][0]['autoscaling_config']['max_replicas'] = 20
    serve_instance.deploy_apps(ServeDeploySchema(**{'applications': [app_config]}))
    print('Redeployed A.')
    wait_for_condition(lambda : check_autoscale_num_replicas(controller, 'A') >= 20, raise_exceptions=True)
    print('Scaled up to 20 requests.')
    second_deployment_replicas = get_running_replica_tags(controller, 'A')
    assert_no_replicas_deprovisioned(first_deployment_replicas, second_deployment_replicas)
    signal.send.remote()
    wait_for_condition(lambda : check_autoscale_num_replicas(controller, 'A') <= 2, raise_exceptions=True)
    assert check_autoscale_num_replicas(controller, 'A') > 1
    assert get_deployment_start_time(controller, 'A') == start_time
    app_config['deployments'][0]['autoscaling_config']['min_replicas'] = 0
    serve_instance.deploy_apps(ServeDeploySchema(**{'applications': [app_config]}))
    print('Redeployed A.')
    wait_for_condition(lambda : get_deployment_status(controller, 'A') == DeploymentStatus.HEALTHY)
    wait_for_condition(lambda : check_autoscale_num_replicas(controller, 'A') < 1, raise_exceptions=True)
    assert check_autoscale_num_replicas(controller, 'A') == 0
    [handle.remote() for _ in range(400)]
    wait_for_condition(lambda : check_autoscale_num_replicas(controller, 'A') > 0, raise_exceptions=True)
    signal.send.remote()
    wait_for_condition(lambda : check_autoscale_num_replicas(controller, 'A') < 1, raise_exceptions=True)

@pytest.mark.skipif(sys.platform == 'win32', reason='Failing on Windows.')
def test_e2e_raise_min_replicas(serve_instance):
    if False:
        i = 10
        return i + 15
    controller = serve_instance._controller
    signal = SignalActor.options(name='signal123').remote()
    app_config = {'import_path': 'ray.serve.tests.test_config_files.get_signal.app', 'deployments': [{'name': 'A', 'autoscaling_config': {'metrics_interval_s': 0.1, 'min_replicas': 0, 'max_replicas': 10, 'look_back_period_s': 0.2, 'downscale_delay_s': 0.2, 'upscale_delay_s': 0.2}, 'graceful_shutdown_timeout_s': 1, 'max_concurrent_queries': 1000}]}
    serve_instance.deploy_apps(ServeDeploySchema(**{'applications': [app_config]}))
    print('Deployed A.')
    wait_for_condition(lambda : get_deployment_status(controller, 'A') == DeploymentStatus.HEALTHY)
    start_time = get_deployment_start_time(controller, 'A')
    assert check_autoscale_num_replicas(controller, 'A') == 0
    handle = serve.get_deployment_handle('A', 'default')
    handle.remote()
    print('Issued one request.')
    time.sleep(2)
    assert check_autoscale_num_replicas(controller, 'A') == 1
    print('Scale up to 1 replica.')
    first_deployment_replicas = get_running_replica_tags(controller, 'A')
    app_config['deployments'][0]['autoscaling_config']['min_replicas'] = 2
    serve_instance.deploy_apps(ServeDeploySchema(**{'applications': [app_config]}))
    print('Redeployed A with min_replicas set to 2.')
    wait_for_condition(lambda : get_deployment_status(controller, 'A') == DeploymentStatus.HEALTHY)
    time.sleep(5)
    assert check_autoscale_num_replicas(controller, 'A') == 2
    print('Autoscaled to 2 without issuing any new requests.')
    second_deployment_replicas = get_running_replica_tags(controller, 'A')
    assert_no_replicas_deprovisioned(first_deployment_replicas, second_deployment_replicas)
    signal.send.remote()
    time.sleep(1)
    print('Completed request.')
    wait_for_condition(lambda : check_autoscale_num_replicas(controller, 'A') <= 2, raise_exceptions=True)
    assert check_autoscale_num_replicas(controller, 'A') > 1
    print('Stayed at 2 replicas.')
    assert get_deployment_start_time(controller, 'A') == start_time

@pytest.mark.skipif(sys.platform == 'win32', reason='Failing on Windows.')
def test_e2e_initial_replicas(serve_instance):
    if False:
        i = 10
        return i + 15

    @serve.deployment(autoscaling_config=AutoscalingConfig(min_replicas=1, initial_replicas=2, max_replicas=5, downscale_delay_s=3))
    def f():
        if False:
            for i in range(10):
                print('nop')
        return os.getpid()
    serve.run(f.bind())
    dep_id = DeploymentID('f', SERVE_DEFAULT_APP_NAME)
    actors = state_api.list_actors(filters=[('class_name', '=', dep_id.to_replica_actor_class_name()), ('state', '=', 'ALIVE')])
    print(actors)
    assert len(actors) == 2

    def check_one_replica():
        if False:
            for i in range(10):
                print('nop')
        actors = state_api.list_actors(filters=[('class_name', '=', dep_id.to_replica_actor_class_name()), ('state', '=', 'ALIVE')])
        return len(actors) == 1
    wait_for_condition(check_one_replica, timeout=20)

@pytest.mark.skipif(sys.platform == 'win32', reason='Failing on Windows.')
def test_e2e_preserve_prev_replicas(serve_instance):
    if False:
        for i in range(10):
            print('nop')
    signal = SignalActor.remote()

    @serve.deployment(max_concurrent_queries=5, autoscaling_config=AutoscalingConfig(min_replicas=1, max_replicas=2, downscale_delay_s=600, upscale_delay_s=0, metrics_interval_s=1, look_back_period_s=1))
    def scaler():
        if False:
            return 10
        ray.get(signal.wait.remote())
        time.sleep(0.2)
        return os.getpid()
    handle = serve.run(scaler.bind())
    dep_id = DeploymentID('scaler', SERVE_DEFAULT_APP_NAME)
    responses = [handle.remote() for _ in range(10)]

    def check_two_replicas():
        if False:
            return 10
        actors = state_api.list_actors(filters=[('class_name', '=', dep_id.to_replica_actor_class_name()), ('state', '=', 'ALIVE')])
        print(actors)
        return len(actors) == 2
    wait_for_condition(check_two_replicas, retry_interval_ms=1000, timeout=20)
    ray.get(signal.send.remote())
    pids = {r.result() for r in responses}
    assert len(pids) == 2
    handle = serve.run(scaler.bind())
    responses = [handle.remote() for _ in range(10)]
    pids = {r.result() for r in responses}
    assert len(pids) == 2

    def check_num_replicas(live: int, dead: int):
        if False:
            for i in range(10):
                print('nop')
        live_actors = state_api.list_actors(filters=[('class_name', '=', dep_id.to_replica_actor_class_name()), ('state', '=', 'ALIVE')])
        dead_actors = state_api.list_actors(filters=[('class_name', '=', dep_id.to_replica_actor_class_name()), ('state', '=', 'DEAD')])
        return len(live_actors) == live and len(dead_actors) == dead
    wait_for_condition(check_num_replicas, retry_interval_ms=1000, timeout=20, live=2, dead=2)
    ray.get(signal.send.remote())
    scaler = scaler.options(autoscaling_config=AutoscalingConfig(min_replicas=1, initial_replicas=3, max_replicas=5, downscale_delay_s=600, upscale_delay_s=600, metrics_interval_s=1, look_back_period_s=1))
    handle = serve.run(scaler.bind())
    responses = [handle.remote() for _ in range(15)]
    pids = {r.result() for r in responses}
    assert len(pids) == 3
    wait_for_condition(check_num_replicas, retry_interval_ms=1000, timeout=20, live=3, dead=4)

@pytest.mark.skipif(sys.platform == 'win32', reason='Failing on Windows.')
def test_e2e_preserve_prev_replicas_rest_api(serve_instance):
    if False:
        return 10
    client = serve_instance
    signal = SignalActor.options(name='signal', namespace='serve').remote()
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_path:
        with zipfile.ZipFile(tmp_path, 'w') as zip_obj:
            with zip_obj.open('app.py', 'w') as f:
                f.write('\nfrom ray import serve\nimport ray\nimport os\n\n@serve.deployment\ndef g():\n    signal = ray.get_actor("signal", namespace="serve")\n    ray.get(signal.wait.remote())\n    return os.getpid()\n\n\napp = g.bind()\n'.encode())
    app_config = {'import_path': 'app:app', 'runtime_env': {'working_dir': f'file://{tmp_path.name}'}, 'deployments': [{'name': 'g', 'autoscaling_config': {'min_replicas': 0, 'max_replicas': 1, 'downscale_delay_s': 600, 'upscale_delay_s': 0, 'metrics_interval_s': 1, 'look_back_period_s': 1}}]}
    client.deploy_apps(ServeDeploySchema(**{'applications': [app_config]}))
    dep_id = DeploymentID('g', SERVE_DEFAULT_APP_NAME)
    wait_for_condition(lambda : serve.status().applications[SERVE_DEFAULT_APP_NAME].status == 'RUNNING')

    @ray.remote
    def send_request():
        if False:
            print('Hello World!')
        return requests.get('http://localhost:8000/').text
    ref = send_request.remote()

    def check_num_replicas(num: int):
        if False:
            print('Hello World!')
        actors = state_api.list_actors(filters=[('class_name', '=', dep_id.to_replica_actor_class_name()), ('state', '=', 'ALIVE')])
        return len(actors) == num
    wait_for_condition(check_num_replicas, retry_interval_ms=1000, timeout=20, num=1)
    signal.send.remote()
    existing_pid = ray.get(ref)
    app_config['deployments'][0]['autoscaling_config']['max_replicas'] = 2
    client.deploy_apps(ServeDeploySchema(**{'applications': [app_config]}))
    wait_for_condition(lambda : serve.status().applications[SERVE_DEFAULT_APP_NAME].status == 'RUNNING')
    wait_for_condition(check_num_replicas, retry_interval_ms=1000, timeout=20, num=1)
    for _ in range(10):
        other_pid = ray.get(send_request.remote())
        assert other_pid == existing_pid
    app_config['deployments'][0]['autoscaling_config']['max_replicas'] = 5
    app_config['deployments'][0]['autoscaling_config']['initial_replicas'] = 3
    app_config['deployments'][0]['autoscaling_config']['upscale_delay'] = 600
    client.deploy_apps(ServeDeploySchema(**{'applications': [app_config]}))
    wait_for_condition(lambda : serve.status().applications[SERVE_DEFAULT_APP_NAME].status == 'RUNNING')
    wait_for_condition(check_num_replicas, retry_interval_ms=1000, timeout=20, num=3)
    pids = set()
    for _ in range(15):
        pids.add(ray.get(send_request.remote()))
    assert existing_pid in pids
if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main(['-v', '-s', __file__]))