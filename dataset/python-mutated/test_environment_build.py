import os
import docker
import pytest
import requests
import socketio
from tests.test_utils import MockRequestReponse, mocked_abortable_async_result, mocked_docker_client, mocked_socketio_class
import app.connections
import app.core.environment_builds
from _orchest.internals.test_utils import raise_exception_function
_NOT_TO_BE_LOGGED = '_NOT_TO_BE_LOGGED'

@pytest.mark.parametrize('image_in_local_environment', [True, False], ids=['image_in_env', 'image_not_in_env'])
@pytest.mark.parametrize('abort', [True, False], ids=['abort_task', 'do_not_abort_task'])
@pytest.mark.parametrize('build_events', [[], [None], ['1', '2', '3', '4'], ['1', '2', '3', '4', None]], ids=['[]', '[None]', '[1, 2, 3, 4]', '[1, 2, 3, 4, None]'])
def test_environment_build(image_in_local_environment, abort, build_events, monkeypatch):
    if False:
        return 10

    def mock_cleanup_docker_artifacts(filters):
        if False:
            i = 10
            return i + 15
        docker_cleanup_uuid_request.add(filters['label'][-1].split('=')[1])

    def mock_put_request(self, url, json=None, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        put_requests.append(json['status'])
        if json['status'] in ['ABORTED', 'SUCCESS', 'FAILURE']:
            monkeypatch.setattr(os, 'fork', lambda : 0)
            monkeypatch.setattr(os, 'kill', lambda *args, **kwargs: True)
        return MockRequestReponse()

    def mock_delete_request(self, url, *args, **kwargs):
        if False:
            print('Hello World!')
        (proj_uuid, env_uuid) = url.split('/')[-2:]
        delete_requests.append((proj_uuid, env_uuid))
        return MockRequestReponse()

    def mock_write_environment_dockerfile(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        pass

    def mock_prepare_build_context(task_uuid, project_uuid, environment_uuid, project_path):
        if False:
            return 10
        return {'snapshot_path': None, 'base_image': None}
    monkeypatch.setattr(requests.sessions.Session, 'put', mock_put_request)
    monkeypatch.setattr(requests.sessions.Session, 'delete', mock_delete_request)
    monkeypatch.setattr(app.core.environment_builds, 'write_environment_dockerfile', mock_write_environment_dockerfile)
    monkeypatch.setattr(app.core.environment_builds, 'prepare_build_context', mock_prepare_build_context)
    monkeypatch.setattr(app.core.environment_builds, '__ENV_BUILD_FULL_LOGS_DIRECTORY', '/tmp/output_environment_build')
    monkeypatch.setattr(app.core.environment_builds, 'cleanup_docker_artifacts', mock_cleanup_docker_artifacts)
    monkeypatch.setattr(app.core.environment_builds, 'AbortableAsyncResult', mocked_abortable_async_result(abort))
    MockedDockerClient = mocked_docker_client(_NOT_TO_BE_LOGGED, build_events)
    if not image_in_local_environment:
        monkeypatch.setattr(MockedDockerClient, 'get', raise_exception_function(docker.errors.ImageNotFound('error')))
    monkeypatch.setattr(app.core.image_utils, 'docker_client', MockedDockerClient())
    socketio_data = {'output_logs': [], 'has_connected': False, 'has_disconnected': False}
    monkeypatch.setattr(socketio, 'Client', mocked_socketio_class(socketio_data))
    put_requests = []
    delete_requests = []
    docker_cleanup_uuid_request = set()
    task_uuid = 'task_uuid'
    project_uuid = ''.join(['events:', str(build_events), '-abort:', str(abort), '-image_in_local_environment:', str(image_in_local_environment)])
    environment_uuid = 'environment_uuid'
    project_path = 'project_path'
    app.core.environment_builds.build_environment_task(task_uuid, project_uuid, environment_uuid, project_path)
    assert len(put_requests) == 2
    assert put_requests[0] == 'STARTED'
    if abort:
        assert put_requests[1] == 'ABORTED'
    elif any([event is None for event in build_events]):
        assert put_requests[1] == 'FAILURE'
    else:
        assert put_requests[1] == 'SUCCESS'
    assert len(delete_requests) == 1
    assert delete_requests[0] == (project_uuid, environment_uuid)
    assert len(docker_cleanup_uuid_request) == 1
    assert task_uuid in docker_cleanup_uuid_request
    assert socketio_data['has_connected']
    assert socketio_data['has_disconnected']
    if not abort:
        if not image_in_local_environment:
            assert 'Pulling image' in socketio_data['output_logs'][0]
        logged_events = ''.join(socketio_data['output_logs'])
        assert _NOT_TO_BE_LOGGED not in logged_events
        expected_events = []
        for event in build_events:
            if event is None:
                break
            expected_events.append(event)
        expected_events = '\n'.join(expected_events)
        assert expected_events in logged_events
    os.remove(os.path.join(app.core.environment_builds.__ENV_BUILD_FULL_LOGS_DIRECTORY, f'orchest-env-{project_uuid}-{environment_uuid}'))