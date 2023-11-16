import sys
from typing import Dict, List, Tuple
from unittest.mock import Mock, PropertyMock, patch
import pytest
from ray.exceptions import RayTaskError
from ray.serve._private.application_state import ApplicationState, ApplicationStateManager, override_deployment_info
from ray.serve._private.common import ApplicationStatus, DeploymentID, DeploymentInfo, DeploymentStatus, DeploymentStatusInfo
from ray.serve._private.config import DeploymentConfig, ReplicaConfig
from ray.serve._private.deploy_utils import deploy_args_to_deployment_info
from ray.serve._private.utils import get_random_letters
from ray.serve.exceptions import RayServeException
from ray.serve.schema import DeploymentSchema, ServeApplicationSchema
from ray.serve.tests.common.utils import MockKVStore

class MockEndpointState:

    def __init__(self):
        if False:
            return 10
        self.endpoints = dict()

    def update_endpoint(self, endpoint, endpoint_info):
        if False:
            print('Hello World!')
        self.endpoints[endpoint] = endpoint_info

    def delete_endpoint(self, endpoint):
        if False:
            i = 10
            return i + 15
        if endpoint in self.endpoints:
            del self.endpoints[endpoint]

class MockDeploymentStateManager:

    def __init__(self, kv_store):
        if False:
            print('Hello World!')
        self.kv_store = kv_store
        self.deployment_infos: Dict[DeploymentID, DeploymentInfo] = dict()
        self.deployment_statuses: Dict[DeploymentID, DeploymentStatusInfo] = dict()
        self.deleting: Dict[DeploymentID, bool] = dict()
        recovered_deployments = self.kv_store.get('fake_deployment_state_checkpoint')
        if recovered_deployments is not None:
            for (name, checkpointed_data) in recovered_deployments.items():
                (info, deleting) = checkpointed_data
                self.deployment_infos[name] = info
                self.deployment_statuses[name] = DeploymentStatus.UPDATING
                self.deleting[name] = deleting

    def deploy(self, deployment_id: DeploymentID, deployment_info: DeploymentInfo):
        if False:
            for i in range(10):
                print('nop')
        existing_info = self.deployment_infos.get(deployment_id)
        self.deleting[deployment_id] = False
        self.deployment_infos[deployment_id] = deployment_info
        if not existing_info or existing_info.version != deployment_info.version:
            self.deployment_statuses[deployment_id] = DeploymentStatusInfo(name=deployment_id.name, status=DeploymentStatus.UPDATING, message='')
        self.kv_store.put('fake_deployment_state_checkpoint', dict(zip(self.deployment_infos.keys(), zip(self.deployment_infos.values(), self.deleting.values()))))

    @property
    def deployments(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        return list(self.deployment_infos.keys())

    def get_deployment_statuses(self, ids: List[DeploymentID]):
        if False:
            for i in range(10):
                print('nop')
        return [self.deployment_statuses[id] for id in ids]

    def get_deployment(self, deployment_id: DeploymentID) -> DeploymentInfo:
        if False:
            print('Hello World!')
        if deployment_id in self.deployment_statuses:
            return DeploymentInfo(deployment_config=DeploymentConfig(num_replicas=1, user_config={}), replica_config=ReplicaConfig.create(lambda x: x), start_time_ms=0, deployer_job_id='')

    def get_deployments_in_application(self, app_name: str):
        if False:
            for i in range(10):
                print('nop')
        deployments = []
        for deployment_id in self.deployment_infos:
            if deployment_id.app == app_name:
                deployments.append(deployment_id.name)
        return deployments

    def set_deployment_unhealthy(self, id: DeploymentID):
        if False:
            while True:
                i = 10
        self.deployment_statuses[id].status = DeploymentStatus.UNHEALTHY

    def set_deployment_healthy(self, id: DeploymentID):
        if False:
            while True:
                i = 10
        self.deployment_statuses[id].status = DeploymentStatus.HEALTHY

    def set_deployment_updating(self, id: DeploymentID):
        if False:
            while True:
                i = 10
        self.deployment_statuses[id].status = DeploymentStatus.UPDATING

    def set_deployment_deleted(self, id: str):
        if False:
            for i in range(10):
                print('nop')
        if not self.deployment_infos[id]:
            raise ValueError(f'Tried to mark deployment {id} as deleted, but {id} not found')
        if not self.deleting[id]:
            raise ValueError(f"Tried to mark deployment {id} as deleted, but delete_deployment()hasn't been called for {id} yet")
        del self.deployment_infos[id]
        del self.deployment_statuses[id]
        del self.deleting[id]

    def delete_deployment(self, id: DeploymentID):
        if False:
            while True:
                i = 10
        self.deleting[id] = True

@pytest.fixture
def mocked_application_state_manager() -> Tuple[ApplicationStateManager, MockDeploymentStateManager]:
    if False:
        return 10
    kv_store = MockKVStore()
    deployment_state_manager = MockDeploymentStateManager(kv_store)
    application_state_manager = ApplicationStateManager(deployment_state_manager, MockEndpointState(), kv_store)
    yield (application_state_manager, deployment_state_manager, kv_store)

def deployment_params(name: str, route_prefix: str=None, docs_path: str=None):
    if False:
        i = 10
        return i + 15
    return {'deployment_name': name, 'deployment_config_proto_bytes': DeploymentConfig(num_replicas=1, user_config={}, version=get_random_letters()).to_proto_bytes(), 'replica_config_proto_bytes': ReplicaConfig.create(lambda x: x).to_proto_bytes(), 'deployer_job_id': 'random', 'route_prefix': route_prefix, 'docs_path': docs_path, 'ingress': False}

def deployment_info(name: str, route_prefix: str=None, docs_path: str=None):
    if False:
        i = 10
        return i + 15
    params = deployment_params(name, route_prefix, docs_path)
    return deploy_args_to_deployment_info(**params, app_name='test_app')

@pytest.fixture
def mocked_application_state() -> Tuple[ApplicationState, MockDeploymentStateManager]:
    if False:
        print('Hello World!')
    kv_store = MockKVStore()
    deployment_state_manager = MockDeploymentStateManager(kv_store)
    application_state = ApplicationState('test_app', deployment_state_manager, MockEndpointState(), lambda *args, **kwargs: None)
    yield (application_state, deployment_state_manager)

@patch.object(ApplicationState, 'target_deployments', PropertyMock(return_value=['a', 'b', 'c']))
class TestDetermineAppStatus:

    @patch.object(ApplicationState, 'get_deployments_statuses')
    def test_running(self, get_deployments_statuses, mocked_application_state):
        if False:
            return 10
        (app_state, _) = mocked_application_state
        get_deployments_statuses.return_value = [DeploymentStatusInfo('a', DeploymentStatus.HEALTHY), DeploymentStatusInfo('b', DeploymentStatus.HEALTHY), DeploymentStatusInfo('c', DeploymentStatus.HEALTHY)]
        assert app_state._determine_app_status() == (ApplicationStatus.RUNNING, '')

    @patch.object(ApplicationState, 'get_deployments_statuses')
    def test_stay_running(self, get_deployments_statuses, mocked_application_state):
        if False:
            print('Hello World!')
        (app_state, _) = mocked_application_state
        app_state._status = ApplicationStatus.RUNNING
        get_deployments_statuses.return_value = [DeploymentStatusInfo('a', DeploymentStatus.HEALTHY), DeploymentStatusInfo('b', DeploymentStatus.HEALTHY), DeploymentStatusInfo('c', DeploymentStatus.HEALTHY)]
        assert app_state._determine_app_status() == (ApplicationStatus.RUNNING, '')

    @patch.object(ApplicationState, 'get_deployments_statuses')
    def test_deploying(self, get_deployments_statuses, mocked_application_state):
        if False:
            i = 10
            return i + 15
        (app_state, _) = mocked_application_state
        get_deployments_statuses.return_value = [DeploymentStatusInfo('a', DeploymentStatus.UPDATING), DeploymentStatusInfo('b', DeploymentStatus.HEALTHY), DeploymentStatusInfo('c', DeploymentStatus.HEALTHY)]
        assert app_state._determine_app_status() == (ApplicationStatus.DEPLOYING, '')

    @patch.object(ApplicationState, 'get_deployments_statuses')
    def test_deploy_failed(self, get_deployments_statuses, mocked_application_state):
        if False:
            while True:
                i = 10
        (app_state, _) = mocked_application_state
        get_deployments_statuses.return_value = [DeploymentStatusInfo('a', DeploymentStatus.UPDATING), DeploymentStatusInfo('b', DeploymentStatus.HEALTHY), DeploymentStatusInfo('c', DeploymentStatus.UNHEALTHY)]
        (status, error_msg) = app_state._determine_app_status()
        assert status == ApplicationStatus.DEPLOY_FAILED
        assert error_msg

    @patch.object(ApplicationState, 'get_deployments_statuses')
    def test_unhealthy(self, get_deployments_statuses, mocked_application_state):
        if False:
            for i in range(10):
                print('nop')
        (app_state, _) = mocked_application_state
        app_state._status = ApplicationStatus.RUNNING
        get_deployments_statuses.return_value = [DeploymentStatusInfo('a', DeploymentStatus.HEALTHY), DeploymentStatusInfo('b', DeploymentStatus.HEALTHY), DeploymentStatusInfo('c', DeploymentStatus.UNHEALTHY)]
        (status, error_msg) = app_state._determine_app_status()
        assert status == ApplicationStatus.UNHEALTHY
        assert error_msg

def test_deploy_and_delete_app(mocked_application_state):
    if False:
        i = 10
        return i + 15
    'Deploy app with 2 deployments, transition DEPLOYING -> RUNNING -> DELETING.\n    This tests the basic typical workflow.\n    '
    (app_state, deployment_state_manager) = mocked_application_state
    d1_id = DeploymentID('d1', 'test_app')
    d2_id = DeploymentID('d2', 'test_app')
    app_state.deploy({'d1': deployment_info('d1', '/hi', '/documentation'), 'd2': deployment_info('d2')})
    assert app_state.route_prefix == '/hi'
    assert app_state.docs_path == '/documentation'
    app_status = app_state.get_application_status_info()
    assert app_status.status == ApplicationStatus.DEPLOYING
    assert app_status.deployment_timestamp > 0
    app_state.update()
    assert deployment_state_manager.get_deployment(d1_id)
    assert deployment_state_manager.get_deployment(d2_id)
    assert app_state.status == ApplicationStatus.DEPLOYING
    app_state.update()
    assert app_state.status == ApplicationStatus.DEPLOYING
    deployment_state_manager.set_deployment_healthy(d1_id)
    app_state.update()
    assert app_state.status == ApplicationStatus.DEPLOYING
    deployment_state_manager.set_deployment_healthy(d2_id)
    app_state.update()
    assert app_state.status == ApplicationStatus.RUNNING
    app_state.update()
    assert app_state.status == ApplicationStatus.RUNNING
    app_state.delete()
    assert app_state.status == ApplicationStatus.DELETING
    app_state.update()
    deployment_state_manager.set_deployment_deleted(d1_id)
    ready_to_be_deleted = app_state.update()
    assert not ready_to_be_deleted
    assert app_state.status == ApplicationStatus.DELETING
    deployment_state_manager.set_deployment_deleted(d2_id)
    ready_to_be_deleted = app_state.update()
    assert ready_to_be_deleted

def test_app_deploy_failed_and_redeploy(mocked_application_state):
    if False:
        for i in range(10):
            print('nop')
    'Test DEPLOYING -> DEPLOY_FAILED -> (redeploy) -> DEPLOYING -> RUNNING'
    (app_state, deployment_state_manager) = mocked_application_state
    d1_id = DeploymentID('d1', 'test_app')
    d2_id = DeploymentID('d2', 'test_app')
    app_state.deploy({'d1': deployment_info('d1')})
    assert app_state.status == ApplicationStatus.DEPLOYING
    app_state.update()
    assert app_state.status == ApplicationStatus.DEPLOYING
    deployment_state_manager.set_deployment_unhealthy(d1_id)
    app_state.update()
    assert app_state.status == ApplicationStatus.DEPLOY_FAILED
    deploy_failed_msg = app_state._status_msg
    assert len(deploy_failed_msg) != 0
    app_state.update()
    assert app_state.status == ApplicationStatus.DEPLOY_FAILED
    assert app_state._status_msg == deploy_failed_msg
    app_state.deploy({'d1': deployment_info('d1'), 'd2': deployment_info('d2')})
    assert app_state.status == ApplicationStatus.DEPLOYING
    assert app_state._status_msg != deploy_failed_msg
    app_state.update()
    assert deployment_state_manager.get_deployment(d1_id)
    assert deployment_state_manager.get_deployment(d2_id)
    assert app_state.status == ApplicationStatus.DEPLOYING
    deployment_state_manager.set_deployment_healthy(d1_id)
    deployment_state_manager.set_deployment_healthy(d2_id)
    app_state.update()
    assert app_state.status == ApplicationStatus.RUNNING
    running_msg = app_state._status_msg
    assert running_msg != deploy_failed_msg
    app_state.update()
    assert app_state.status == ApplicationStatus.RUNNING
    assert app_state._status_msg == running_msg

def test_app_deploy_failed_and_recover(mocked_application_state):
    if False:
        while True:
            i = 10
    'Test DEPLOYING -> DEPLOY_FAILED -> (self recovered) -> RUNNING\n\n    If while the application is deploying a deployment becomes unhealthy,\n    the app is marked as deploy failed. But if the deployment recovers,\n    the application status should update to running.\n    '
    (app_state, deployment_state_manager) = mocked_application_state
    deployment_id = DeploymentID('d1', 'test_app')
    app_state.deploy({'d1': deployment_info('d1')})
    assert app_state.status == ApplicationStatus.DEPLOYING
    app_state.update()
    assert app_state.status == ApplicationStatus.DEPLOYING
    deployment_state_manager.set_deployment_unhealthy(deployment_id)
    app_state.update()
    assert app_state.status == ApplicationStatus.DEPLOY_FAILED
    app_state.update()
    assert app_state.status == ApplicationStatus.DEPLOY_FAILED
    deployment_state_manager.set_deployment_healthy(deployment_id)
    app_state.update()
    assert app_state.status == ApplicationStatus.RUNNING
    app_state.update()
    assert app_state.status == ApplicationStatus.RUNNING

def test_app_unhealthy(mocked_application_state):
    if False:
        for i in range(10):
            print('nop')
    'Test DEPLOYING -> RUNNING -> UNHEALTHY -> RUNNING.\n    Even after an application becomes running, if a deployment becomes\n    unhealthy at some point, the application status should also be\n    updated to unhealthy.\n    '
    (app_state, deployment_state_manager) = mocked_application_state
    (id_a, id_b) = (DeploymentID('a', 'test_app'), DeploymentID('b', 'test_app'))
    app_state.deploy({'a': deployment_info('a'), 'b': deployment_info('b')})
    assert app_state.status == ApplicationStatus.DEPLOYING
    app_state.update()
    assert app_state.status == ApplicationStatus.DEPLOYING
    deployment_state_manager.set_deployment_healthy(id_a)
    deployment_state_manager.set_deployment_healthy(id_b)
    app_state.update()
    assert app_state.status == ApplicationStatus.RUNNING
    deployment_state_manager.set_deployment_unhealthy(id_a)
    app_state.update()
    assert app_state.status == ApplicationStatus.UNHEALTHY
    app_state.update()
    assert app_state.status == ApplicationStatus.UNHEALTHY
    deployment_state_manager.set_deployment_healthy(id_a)
    app_state.update()
    assert app_state.status == ApplicationStatus.RUNNING

@patch('ray.serve._private.application_state.build_serve_application', Mock())
@patch('ray.get', Mock(return_value=([deployment_params('a', '/old', '/docs')], None)))
@patch('ray.serve._private.application_state.check_obj_ref_ready_nowait')
def test_deploy_through_config_succeed(check_obj_ref_ready_nowait):
    if False:
        for i in range(10):
            print('nop')
    'Test deploying through config successfully.\n    Deploy obj ref finishes successfully, so status should transition to running.\n    '
    kv_store = MockKVStore()
    deployment_id = DeploymentID('a', 'test_app')
    deployment_state_manager = MockDeploymentStateManager(kv_store)
    app_state_manager = ApplicationStateManager(deployment_state_manager, MockEndpointState(), kv_store)
    app_config = ServeApplicationSchema(import_path='fa.ke', route_prefix='/new')
    app_state_manager.deploy_config(name='test_app', app_config=app_config)
    app_state = app_state_manager._application_states['test_app']
    assert app_state.status == ApplicationStatus.DEPLOYING
    check_obj_ref_ready_nowait.return_value = False
    app_state.update()
    assert app_state._build_app_task_info
    assert app_state.status == ApplicationStatus.DEPLOYING
    app_state.update()
    assert app_state.status == ApplicationStatus.DEPLOYING
    check_obj_ref_ready_nowait.return_value = True
    app_state.update()
    assert app_state.status == ApplicationStatus.DEPLOYING
    assert app_state.target_deployments == ['a']
    assert app_state.route_prefix == '/new'
    assert app_state.docs_path == '/docs'
    deployment_state_manager.set_deployment_healthy(deployment_id)
    app_state.update()
    assert app_state.status == ApplicationStatus.RUNNING

@patch('ray.serve._private.application_state.get_app_code_version', Mock(return_value='123'))
@patch('ray.serve._private.application_state.build_serve_application', Mock())
@patch('ray.get', Mock(side_effect=RayTaskError(None, 'intentionally failed', None)))
@patch('ray.serve._private.application_state.check_obj_ref_ready_nowait')
def test_deploy_through_config_fail(check_obj_ref_ready_nowait):
    if False:
        print('Hello World!')
    'Test fail to deploy through config.\n    Deploy obj ref errors out, so status should transition to deploy failed.\n    '
    kv_store = MockKVStore()
    deployment_state_manager = MockDeploymentStateManager(kv_store)
    app_state_manager = ApplicationStateManager(deployment_state_manager, MockEndpointState(), kv_store)
    app_state_manager.deploy_config(name='test_app', app_config=Mock())
    app_state = app_state_manager._application_states['test_app']
    assert app_state.status == ApplicationStatus.DEPLOYING
    check_obj_ref_ready_nowait.return_value = False
    app_state.update()
    assert app_state._build_app_task_info
    assert app_state.status == ApplicationStatus.DEPLOYING
    app_state.update()
    assert app_state.status == ApplicationStatus.DEPLOYING
    check_obj_ref_ready_nowait.return_value = True
    app_state.update()
    assert app_state.status == ApplicationStatus.DEPLOY_FAILED
    assert 'failed' in app_state._status_msg or 'error' in app_state._status_msg

def test_redeploy_same_app(mocked_application_state):
    if False:
        while True:
            i = 10
    'Test redeploying same application with updated deployments.'
    (app_state, deployment_state_manager) = mocked_application_state
    a_id = DeploymentID('a', 'test_app')
    b_id = DeploymentID('b', 'test_app')
    c_id = DeploymentID('c', 'test_app')
    app_state.deploy({'a': deployment_info('a'), 'b': deployment_info('b')})
    assert app_state.status == ApplicationStatus.DEPLOYING
    app_state.update()
    assert app_state.status == ApplicationStatus.DEPLOYING
    assert set(app_state.target_deployments) == {'a', 'b'}
    deployment_state_manager.set_deployment_healthy(a_id)
    app_state.update()
    assert app_state.status == ApplicationStatus.DEPLOYING
    deployment_state_manager.set_deployment_healthy(b_id)
    app_state.update()
    assert app_state.status == ApplicationStatus.RUNNING
    app_state.deploy({'b': deployment_info('b'), 'c': deployment_info('c')})
    assert app_state.status == ApplicationStatus.DEPLOYING
    assert 'a' not in app_state.target_deployments
    app_state.update()
    deployment_state_manager.set_deployment_deleted(a_id)
    app_state.update()
    assert app_state.status == ApplicationStatus.DEPLOYING
    deployment_state_manager.set_deployment_healthy(c_id)
    app_state.update()
    assert app_state.status == ApplicationStatus.DEPLOYING
    deployment_state_manager.set_deployment_healthy(b_id)
    app_state.update()
    assert app_state.status == ApplicationStatus.RUNNING

def test_deploy_with_route_prefix_conflict(mocked_application_state_manager):
    if False:
        while True:
            i = 10
    'Test that an application with a route prefix conflict fails to deploy'
    (app_state_manager, _, _) = mocked_application_state_manager
    app_state_manager.apply_deployment_args('app1', [deployment_params('a', '/hi')])
    with pytest.raises(RayServeException):
        app_state_manager.apply_deployment_args('app2', [deployment_params('b', '/hi')])

def test_deploy_with_renamed_app(mocked_application_state_manager):
    if False:
        print('Hello World!')
    '\n    Test that an application deploys successfully when there is a route prefix\n    conflict with an old app running on the cluster.\n    '
    (app_state_manager, deployment_state_manager, _) = mocked_application_state_manager
    (a_id, b_id) = (DeploymentID('a', 'app1'), DeploymentID('b', 'app2'))
    app_state_manager.apply_deployment_args('app1', [deployment_params('a', '/url1')])
    app_state = app_state_manager._application_states['app1']
    assert app_state_manager.get_app_status('app1') == ApplicationStatus.DEPLOYING
    app_state_manager.update()
    assert app_state_manager.get_app_status('app1') == ApplicationStatus.DEPLOYING
    assert set(app_state.target_deployments) == {'a'}
    deployment_state_manager.set_deployment_healthy(a_id)
    app_state_manager.update()
    assert app_state_manager.get_app_status('app1') == ApplicationStatus.RUNNING
    app_state_manager.delete_application('app1')
    assert app_state_manager.get_app_status('app1') == ApplicationStatus.DELETING
    app_state_manager.update()
    app_state_manager.apply_deployment_args('app2', [deployment_params('b', '/url1')])
    assert app_state_manager.get_app_status('app2') == ApplicationStatus.DEPLOYING
    app_state_manager.update()
    deployment_state_manager.set_deployment_healthy(b_id)
    app_state_manager.update()
    assert app_state_manager.get_app_status('app2') == ApplicationStatus.RUNNING
    assert app_state_manager.get_app_status('app1') == ApplicationStatus.DELETING
    deployment_state_manager.set_deployment_deleted(a_id)
    app_state_manager.update()
    assert app_state_manager.get_app_status('app1') == ApplicationStatus.NOT_STARTED
    assert app_state_manager.get_app_status('app2') == ApplicationStatus.RUNNING

def test_application_state_recovery(mocked_application_state_manager):
    if False:
        while True:
            i = 10
    'Test DEPLOYING -> RUNNING -> (controller crash) -> DEPLOYING -> RUNNING'
    (app_state_manager, deployment_state_manager, kv_store) = mocked_application_state_manager
    deployment_id = DeploymentID('d1', 'test_app')
    app_name = 'test_app'
    params = deployment_params('d1')
    app_state_manager.apply_deployment_args(app_name, [params])
    app_state = app_state_manager._application_states[app_name]
    assert app_state.status == ApplicationStatus.DEPLOYING
    app_state_manager.update()
    assert deployment_state_manager.get_deployment(deployment_id)
    deployment_state_manager.set_deployment_healthy(deployment_id)
    app_state_manager.update()
    assert app_state.status == ApplicationStatus.RUNNING
    new_deployment_state_manager = MockDeploymentStateManager(kv_store)
    version1 = new_deployment_state_manager.deployment_infos[deployment_id].version
    new_app_state_manager = ApplicationStateManager(new_deployment_state_manager, MockEndpointState(), kv_store)
    app_state = new_app_state_manager._application_states[app_name]
    assert app_state.status == ApplicationStatus.DEPLOYING
    assert app_state._target_state.deployment_infos['d1'].version == version1
    new_deployment_state_manager.set_deployment_healthy(deployment_id)
    new_app_state_manager.update()
    assert app_state.status == ApplicationStatus.RUNNING

def test_recover_during_update(mocked_application_state_manager):
    if False:
        print('Hello World!')
    'Test that application and deployment states are recovered if\n    controller crashed in the middle of a redeploy.\n\n    Target state is checkpointed in the application state manager,\n    but not yet the deployment state manager when the controller crashes\n    Then the deployment state manager should recover an old version of\n    the deployment during initial recovery, but the application state\n    manager should eventually reconcile this.\n    '
    (app_state_manager, deployment_state_manager, kv_store) = mocked_application_state_manager
    deployment_id = DeploymentID('d1', 'test_app')
    app_name = 'test_app'
    params = deployment_params('d1')
    app_state_manager.apply_deployment_args(app_name, [params])
    app_state = app_state_manager._application_states[app_name]
    assert app_state.status == ApplicationStatus.DEPLOYING
    app_state_manager.update()
    assert deployment_state_manager.get_deployment(deployment_id)
    deployment_state_manager.set_deployment_healthy(deployment_id)
    app_state_manager.update()
    assert app_state.status == ApplicationStatus.RUNNING
    params2 = deployment_params('d1')
    app_state_manager.apply_deployment_args(app_name, [params2])
    assert app_state.status == ApplicationStatus.DEPLOYING
    new_deployment_state_manager = MockDeploymentStateManager(kv_store)
    dr_version = new_deployment_state_manager.deployment_infos[deployment_id].version
    new_app_state_manager = ApplicationStateManager(new_deployment_state_manager, MockEndpointState(), kv_store)
    app_state = new_app_state_manager._application_states[app_name]
    ar_version = app_state._target_state.deployment_infos['d1'].version
    assert app_state.status == ApplicationStatus.DEPLOYING
    assert ar_version != dr_version
    new_app_state_manager.update()
    assert new_deployment_state_manager.deployment_infos[deployment_id].version == ar_version
    assert app_state.status == ApplicationStatus.DEPLOYING
    new_deployment_state_manager.set_deployment_healthy(deployment_id)
    new_app_state_manager.update()
    assert app_state.status == ApplicationStatus.RUNNING

def test_is_ready_for_shutdown(mocked_application_state_manager):
    if False:
        while True:
            i = 10
    'Test `is_ready_for_shutdown()` returns the correct state.\n\n    When shutting down applications before deployments are deleted, application state\n    `is_deleted()` should return False and `is_ready_for_shutdown()` should return\n    False. When shutting down applications after deployments are deleted, application\n    state `is_deleted()` should return True and `is_ready_for_shutdown()` should return\n    True.\n    '
    (app_state_manager, deployment_state_manager, kv_store) = mocked_application_state_manager
    app_name = 'test_app'
    deployment_name = 'd1'
    deployment_id = DeploymentID(deployment_name, app_name)
    params = deployment_params(deployment_name)
    app_state_manager.apply_deployment_args(app_name, [params])
    app_state = app_state_manager._application_states[app_name]
    assert app_state.status == ApplicationStatus.DEPLOYING
    app_state_manager.update()
    assert deployment_state_manager.get_deployment(deployment_id)
    deployment_state_manager.set_deployment_healthy(deployment_id)
    app_state_manager.update()
    assert app_state.status == ApplicationStatus.RUNNING
    app_state_manager.shutdown()
    assert not app_state.is_deleted()
    assert not app_state_manager.is_ready_for_shutdown()
    deployment_state_manager.delete_deployment(deployment_id)
    deployment_state_manager.set_deployment_deleted(deployment_id)
    app_state_manager.update()
    assert app_state.is_deleted()
    assert app_state_manager.is_ready_for_shutdown()

class TestOverrideDeploymentInfo:

    @pytest.fixture
    def info(self):
        if False:
            for i in range(10):
                print('nop')
        return DeploymentInfo(route_prefix='/', version='123', deployment_config=DeploymentConfig(num_replicas=1), replica_config=ReplicaConfig.create(lambda x: x), start_time_ms=0, deployer_job_id='')

    def test_override_deployment_config(self, info):
        if False:
            while True:
                i = 10
        config = ServeApplicationSchema(name='default', import_path='test.import.path', deployments=[DeploymentSchema(name='A', num_replicas=3, max_concurrent_queries=200, user_config={'price': '4'}, graceful_shutdown_wait_loop_s=4, graceful_shutdown_timeout_s=40, health_check_period_s=20, health_check_timeout_s=60)])
        updated_infos = override_deployment_info('default', {'A': info}, config)
        updated_info = updated_infos['A']
        assert updated_info.route_prefix == '/'
        assert updated_info.version == '123'
        assert updated_info.deployment_config.max_concurrent_queries == 200
        assert updated_info.deployment_config.user_config == {'price': '4'}
        assert updated_info.deployment_config.graceful_shutdown_wait_loop_s == 4
        assert updated_info.deployment_config.graceful_shutdown_timeout_s == 40
        assert updated_info.deployment_config.health_check_period_s == 20
        assert updated_info.deployment_config.health_check_timeout_s == 60

    def test_override_autoscaling_config(self, info):
        if False:
            while True:
                i = 10
        config = ServeApplicationSchema(name='default', import_path='test.import.path', deployments=[DeploymentSchema(name='A', autoscaling_config={'min_replicas': 1, 'initial_replicas': 12, 'max_replicas': 79})])
        updated_infos = override_deployment_info('default', {'A': info}, config)
        updated_info = updated_infos['A']
        assert updated_info.route_prefix == '/'
        assert updated_info.version == '123'
        assert updated_info.autoscaling_policy.config.min_replicas == 1
        assert updated_info.autoscaling_policy.config.initial_replicas == 12
        assert updated_info.autoscaling_policy.config.max_replicas == 79

    def test_override_route_prefix_1(self, info):
        if False:
            print('Hello World!')
        config = ServeApplicationSchema(name='default', import_path='test.import.path', deployments=[DeploymentSchema(name='A', route_prefix='/alice')])
        updated_infos = override_deployment_info('default', {'A': info}, config)
        updated_info = updated_infos['A']
        assert updated_info.route_prefix == '/alice'
        assert updated_info.version == '123'

    def test_override_route_prefix_2(self, info):
        if False:
            for i in range(10):
                print('nop')
        config = ServeApplicationSchema(name='default', import_path='test.import.path', route_prefix='/bob', deployments=[DeploymentSchema(name='A')])
        updated_infos = override_deployment_info('default', {'A': info}, config)
        updated_info = updated_infos['A']
        assert updated_info.route_prefix == '/bob'
        assert updated_info.version == '123'

    def test_override_route_prefix_3(self, info):
        if False:
            print('Hello World!')
        config = ServeApplicationSchema(name='default', import_path='test.import.path', route_prefix='/bob', deployments=[DeploymentSchema(name='A', route_prefix='/alice')])
        updated_infos = override_deployment_info('default', {'A': info}, config)
        updated_info = updated_infos['A']
        assert updated_info.route_prefix == '/bob'
        assert updated_info.version == '123'

    def test_override_ray_actor_options_1(self, info):
        if False:
            while True:
                i = 10
        'Test runtime env specified in config at deployment level.'
        config = ServeApplicationSchema(name='default', import_path='test.import.path', deployments=[DeploymentSchema(name='A', ray_actor_options={'runtime_env': {'working_dir': 's3://B'}})])
        updated_infos = override_deployment_info('default', {'A': info}, config)
        updated_info = updated_infos['A']
        assert updated_info.route_prefix == '/'
        assert updated_info.version == '123'
        assert updated_info.replica_config.ray_actor_options['runtime_env']['working_dir'] == 's3://B'

    def test_override_ray_actor_options_2(self, info):
        if False:
            i = 10
            return i + 15
        'Test application runtime env is propagated to deployments.'
        config = ServeApplicationSchema(name='default', import_path='test.import.path', runtime_env={'working_dir': 's3://C'}, deployments=[DeploymentSchema(name='A')])
        updated_infos = override_deployment_info('default', {'A': info}, config)
        updated_info = updated_infos['A']
        assert updated_info.route_prefix == '/'
        assert updated_info.version == '123'
        assert updated_info.replica_config.ray_actor_options['runtime_env']['working_dir'] == 's3://C'

    def test_override_ray_actor_options_3(self, info):
        if False:
            for i in range(10):
                print('nop')
        'If runtime env is specified in the config at the deployment level, it should\n        override the application-level runtime env.\n        '
        config = ServeApplicationSchema(name='default', import_path='test.import.path', runtime_env={'working_dir': 's3://C'}, deployments=[DeploymentSchema(name='A', ray_actor_options={'runtime_env': {'working_dir': 's3://B'}})])
        updated_infos = override_deployment_info('default', {'A': info}, config)
        updated_info = updated_infos['A']
        assert updated_info.route_prefix == '/'
        assert updated_info.version == '123'
        assert updated_info.replica_config.ray_actor_options['runtime_env']['working_dir'] == 's3://B'

    def test_override_ray_actor_options_4(self):
        if False:
            while True:
                i = 10
        'If runtime env is specified for the deployment in code, it should override\n        the application-level runtime env.\n        '
        info = DeploymentInfo(route_prefix='/', version='123', deployment_config=DeploymentConfig(num_replicas=1), replica_config=ReplicaConfig.create(lambda x: x, ray_actor_options={'runtime_env': {'working_dir': 's3://A'}}), start_time_ms=0, deployer_job_id='')
        config = ServeApplicationSchema(name='default', import_path='test.import.path', runtime_env={'working_dir': 's3://C'}, deployments=[DeploymentSchema(name='A')])
        updated_infos = override_deployment_info('default', {'A': info}, config)
        updated_info = updated_infos['A']
        assert updated_info.route_prefix == '/'
        assert updated_info.version == '123'
        assert updated_info.replica_config.ray_actor_options['runtime_env']['working_dir'] == 's3://A'

    def test_override_ray_actor_options_5(self):
        if False:
            return 10
        'If runtime env is specified in all three places:\n        - In code\n        - In the config at the deployment level\n        - In the config at the application level\n        The one specified in the config at the deployment level should take precedence.\n        '
        info = DeploymentInfo(route_prefix='/', version='123', deployment_config=DeploymentConfig(num_replicas=1), replica_config=ReplicaConfig.create(lambda x: x, ray_actor_options={'runtime_env': {'working_dir': 's3://A'}}), start_time_ms=0, deployer_job_id='')
        config = ServeApplicationSchema(name='default', import_path='test.import.path', runtime_env={'working_dir': 's3://C'}, deployments=[DeploymentSchema(name='A', ray_actor_options={'runtime_env': {'working_dir': 's3://B'}})])
        updated_infos = override_deployment_info('default', {'A': info}, config)
        updated_info = updated_infos['A']
        assert updated_info.route_prefix == '/'
        assert updated_info.version == '123'
        assert updated_info.replica_config.ray_actor_options['runtime_env']['working_dir'] == 's3://B'
if __name__ == '__main__':
    sys.exit(pytest.main(['-v', '-s', __file__]))