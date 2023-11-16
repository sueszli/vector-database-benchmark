import docker
from app.apis import namespace_validations as ns_val

def test_gate_post_success(client, monkeypatch):
    if False:
        print('Hello World!')

    class DockerClient:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.images = self

        def get(*args, **kwargs):
            if False:
                while True:
                    i = 10
            return []
    monkeypatch.setattr(ns_val, 'docker_client', DockerClient())
    req = {'project_uuid': '1', 'environment_uuids': ['1', '2']}
    resp = client.post('/api/validations/environments', json=req)
    data = resp.get_json()
    assert resp.status_code == 201
    assert data['validation'] == 'pass'
    assert data['pass'] == ['1', '2']

def test_gate_post_api_error(client, monkeypatch):
    if False:
        return 10

    class DockerClient:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.images = self

        def get(*args, **kwargs):
            if False:
                return 10
            raise docker.errors.APIError('')
    monkeypatch.setattr(ns_val, 'docker_client', DockerClient())
    req = {'project_uuid': '1', 'environment_uuids': ['1', '2']}
    resp = client.post('/api/validations/environments', json=req)
    data = resp.get_json()
    assert resp.status_code == 201
    assert data['validation'] == 'fail'
    assert data['fail'] == ['1', '2']
    assert data['actions'] == ['RETRY', 'RETRY']

def test_gate_post_image_not_found_wait(client, environment_build, monkeypatch):
    if False:
        i = 10
        return i + 15

    class DockerClient:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.images = self

        def get(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            raise docker.errors.ImageNotFound('')
    monkeypatch.setattr(ns_val, 'docker_client', DockerClient())
    req = {'project_uuid': environment_build.project.uuid, 'environment_uuids': [environment_build.environment_uuid]}
    resp = client.post('/api/validations/environments', json=req)
    data = resp.get_json()
    assert resp.status_code == 201
    assert data['validation'] == 'fail'
    assert data['fail'] == [environment_build.environment_uuid]
    assert data['actions'] == ['WAIT']

def test_gate_post_image_not_found_build(client, monkeypatch):
    if False:
        print('Hello World!')

    class DockerClient:

        def __init__(self):
            if False:
                return 10
            self.images = self

        def get(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            raise docker.errors.ImageNotFound('')
    monkeypatch.setattr(ns_val, 'docker_client', DockerClient())
    req = {'project_uuid': '1', 'environment_uuids': ['1']}
    resp = client.post('/api/validations/environments', json=req)
    data = resp.get_json()
    assert resp.status_code == 201
    assert data['validation'] == 'fail'
    assert data['fail'] == ['1']
    assert data['actions'] == ['BUILD']