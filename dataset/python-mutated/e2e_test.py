from collections.abc import Iterator
import os
import subprocess
import uuid
import backoff
import pytest
import requests
SUFFIX = uuid.uuid4().hex[:10]
SAMPLE_VERSION = os.environ.get('SAMPLE_VERSION', None)
GOOGLE_CLOUD_PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']
REGION = 'us-central1'
PLATFORM = 'managed'
SERVICE = f'polls-{SUFFIX}'
POSTGRES_INSTANCE = os.environ.get('POSTGRES_INSTANCE', None)
if not POSTGRES_INSTANCE:
    raise Exception("'POSTGRES_INSTANCE' env var not found")
if ':' in POSTGRES_INSTANCE:
    POSTGRES_INSTANCE_FULL = POSTGRES_INSTANCE
    POSTGRES_INSTANCE_NAME = POSTGRES_INSTANCE.split(':')[-1]
else:
    POSTGRES_INSTANCE_FULL = f'{GOOGLE_CLOUD_PROJECT}:{REGION}:{POSTGRES_INSTANCE}'
    POSTGRES_INSTANCE_NAME = POSTGRES_INSTANCE
POSTGRES_DATABASE = f'django-database-{SUFFIX}'
CLOUD_STORAGE_BUCKET = f'{GOOGLE_CLOUD_PROJECT}-media-{SUFFIX}'
POSTGRES_DATABASE = f'polls-{SUFFIX}'
POSTGRES_USER = f'django-{SUFFIX}'
POSTGRES_PASSWORD = uuid.uuid4().hex[:26]
ADMIN_NAME = 'admin'
ADMIN_PASSWORD = uuid.uuid4().hex[:26]
SECRET_SETTINGS_NAME = f'django_settings-{SUFFIX}'
SECRET_PASSWORD_NAME = f'superuser_password-{SUFFIX}'

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def run_shell_cmd(args: list) -> subprocess.CompletedProcess:
    if False:
        print('Hello World!')
    '\n    Runs a shell command and returns its output.\n    Usage: run_shell_cmd(args)\n        args: an array of command line arguments\n    Example:\n        result = run_shell_command(["gcloud, "app", "deploy"])\n        print("The command\'s stdout was:", result.stdout)\n\n    Raises Exception with the stderr output of the last attempt on failure.\n    '
    full_command = ' '.join(args)
    print('Running command:', full_command)
    try:
        output = subprocess.run(full_command, capture_output=True, shell=True, check=True)
        return output
    except subprocess.CalledProcessError as e:
        print('Command failed')
        print(f'stderr was {e.stderr}')
        raise e

@pytest.fixture
def deployed_service() -> str:
    if False:
        for i in range(10):
            print('nop')
    substitutions = [f'_SERVICE={SERVICE},_PLATFORM={PLATFORM},_REGION={REGION},_STORAGE_BUCKET={CLOUD_STORAGE_BUCKET},_DB_NAME={POSTGRES_DATABASE},_DB_USER={POSTGRES_USER},_DB_PASS={POSTGRES_PASSWORD},_DB_INSTANCE={POSTGRES_INSTANCE_NAME},_SECRET_SETTINGS_NAME={SECRET_SETTINGS_NAME},_SECRET_PASSWORD_NAME={SECRET_PASSWORD_NAME},_SECRET_PASSWORD_VALUE={ADMIN_PASSWORD},_CLOUD_SQL_CONNECTION_NAME={POSTGRES_INSTANCE_FULL}']
    if SAMPLE_VERSION:
        substitutions.append(f',_VERSION={SAMPLE_VERSION}')
    run_shell_cmd(['gcloud', 'builds', 'submit', '--project', GOOGLE_CLOUD_PROJECT, '--config', './e2e_test_setup.yaml', '--substitutions'] + substitutions)
    yield SERVICE
    substitutions = [f'_SERVICE={SERVICE},_PLATFORM={PLATFORM},_REGION={REGION},_DB_USER={POSTGRES_USER},_DB_NAME={POSTGRES_DATABASE},_DB_INSTANCE={POSTGRES_INSTANCE_NAME},_SECRET_SETTINGS_NAME={SECRET_SETTINGS_NAME},_SECRET_PASSWORD_NAME={SECRET_PASSWORD_NAME},_STORAGE_BUCKET={CLOUD_STORAGE_BUCKET},']
    if SAMPLE_VERSION:
        substitutions.append(f'_SAMPLE_VERSION={SAMPLE_VERSION}')
    run_shell_cmd(['gcloud', 'builds', 'submit', '--project', GOOGLE_CLOUD_PROJECT, '--config', './e2e_test_cleanup.yaml', '--substitutions'] + substitutions)

@pytest.fixture
def service_url_auth_token(deployed_service: str) -> Iterator[tuple[str, str]]:
    if False:
        print('Hello World!')
    service_url = run_shell_cmd(['gcloud', 'run', 'services', 'describe', deployed_service, '--platform', 'managed', '--region', REGION, '--format', '"value(status.url)"', '--project', GOOGLE_CLOUD_PROJECT]).stdout.strip().decode()
    auth_token = run_shell_cmd(['gcloud', 'auth', 'print-identity-token', '--project', GOOGLE_CLOUD_PROJECT]).stdout.strip().decode()
    yield (service_url, auth_token)

def test_end_to_end(service_url_auth_token: list[str]) -> None:
    if False:
        print('Hello World!')
    (service_url, auth_token) = service_url_auth_token
    headers = {'Authorization': f'Bearer {auth_token}'}
    login_slug = '/admin/login/?next=/admin/'
    client = requests.session()
    response = client.get(service_url, headers=headers)
    body = response.text
    assert response.status_code == 200
    assert 'Hello, world' in body
    client.get(service_url + login_slug, headers=headers)
    csrftoken = client.cookies['csrftoken']
    payload = {'username': ADMIN_NAME, 'password': ADMIN_PASSWORD, 'csrfmiddlewaretoken': csrftoken}
    response = client.post(service_url + login_slug, data=payload, headers=headers)
    body = response.text
    assert response.status_code == 200
    assert 'Please enter the correct username and password' not in body
    assert 'Site administration' in body
    assert 'Polls' in body