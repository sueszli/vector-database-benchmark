import os
import subprocess
from urllib import request
import uuid
import pytest

@pytest.fixture()
def services():
    if False:
        for i in range(10):
            print('nop')
    suffix = uuid.uuid4().hex
    project = os.environ['GOOGLE_CLOUD_PROJECT']
    subprocess.run(['gcloud', 'run', 'deploy', f'helloworld-{suffix}', '--project', project, '--image=gcr.io/cloudrun/hello', '--platform=managed', '--region=us-central1', '--no-allow-unauthenticated', '--quiet'], check=True)
    endpoint = subprocess.run(['gcloud', 'run', 'services', 'describe', f'helloworld-{suffix}', '--project', project, '--platform=managed', '--region=us-central1', '--format=value(status.url)'], stdout=subprocess.PIPE, check=True).stdout.strip()
    subprocess.run(['gcloud', 'functions', 'deploy', f'helloworld-{suffix}', '--project', project, '--runtime=python38', '--region=us-central1', '--trigger-http', '--no-allow-unauthenticated', '--entry-point=get_authorized', f'--set-env-vars=URL={endpoint.decode()}'], check=True)
    function_url = f'https://us-central1-{project}.cloudfunctions.net/helloworld-{suffix}'
    token = subprocess.run(['gcloud', 'auth', 'print-identity-token'], stdout=subprocess.PIPE, check=True).stdout.strip()
    yield (function_url, token)
    subprocess.run(['gcloud', 'run', 'services', 'delete', f'helloworld-{suffix}', '--project', project, '--async', '--platform=managed', '--region=us-central1', '--quiet'], check=True)
    subprocess.run(['gcloud', 'functions', 'delete', f'helloworld-{suffix}', '--project', project, '--region=us-central1', '--quiet'], check=True)

def test_auth(services):
    if False:
        while True:
            i = 10
    url = services[0]
    token = services[1].decode()
    req = request.Request(url, headers={'Authorization': f'Bearer {token}'})
    response = request.urlopen(req)
    assert response.status == 200
    assert 'Hello World' in response.read().decode()