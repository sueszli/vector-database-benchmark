import enum
import time
import pytest
import requests
import requests.exceptions
from saltfactories.utils import random_string
from salt.utils.etcd_util import HAS_ETCD_V2, HAS_ETCD_V3

class EtcdVersion(enum.Enum):
    v2 = 'etcd-v2'
    v3 = 'etcd-v3'
    v3_v2_mode = 'etcd-v3(v2-mode)'

def etcd_version_ids(enum_value):
    if False:
        while True:
            i = 10
    return enum_value.value

@pytest.fixture(scope='module', params=tuple(EtcdVersion), ids=etcd_version_ids)
def etcd_version(request):
    if False:
        for i in range(10):
            print('nop')
    if request.param == EtcdVersion.v2 and (not HAS_ETCD_V2):
        pytest.skip('No etcd library installed')
    if request.param != EtcdVersion.v2 and (not HAS_ETCD_V3):
        pytest.skip('No etcd3 library installed')
    return request.param

@pytest.fixture(scope='module')
def etcd_container_image_name(etcd_version):
    if False:
        while True:
            i = 10
    if etcd_version == EtcdVersion.v2:
        return 'ghcr.io/saltstack/salt-ci-containers/etcd:2'
    return 'ghcr.io/saltstack/salt-ci-containers/etcd:3'

@pytest.fixture(scope='module')
def etcd_container_name(etcd_version):
    if False:
        return 10
    if etcd_version == EtcdVersion.v2:
        return random_string('etcd-v2-server-')
    if etcd_version == EtcdVersion.v3:
        return random_string('etcd-v3-server-')
    return random_string('etcd-v3-in-v2-mode-server-')

@pytest.fixture(scope='module')
def etcd_static_port():
    if False:
        while True:
            i = 10
    '\n    We return ``None`` because we want docker to assign the host port for us.\n\n    Return a port number to override the above behavior.\n    '
    return None

def confirm_container_started(timeout_at, container):
    if False:
        i = 10
        return i + 15
    etcd_port = container.get_host_port_binding(2379, protocol='tcp', ipv6=False)
    sleeptime = 1
    while time.time() <= timeout_at:
        try:
            response = requests.get('http://localhost:{}/version'.format(etcd_port))
            try:
                version = response.json()
                if 'etcdserver' in version:
                    break
            except ValueError:
                if 'etcd 2.' in response.text:
                    break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(sleeptime)
        sleeptime *= 2
    else:
        return False
    return True

@pytest.fixture(scope='module')
def etcd_container(salt_factories, etcd_container_name, etcd_container_image_name, etcd_static_port, etcd_version):
    if False:
        i = 10
        return i + 15
    container_environment = {'ALLOW_NONE_AUTHENTICATION': 'yes'}
    if etcd_version == EtcdVersion.v3_v2_mode:
        container_environment['ETCD_ENABLE_V2'] = 'true'
    container = salt_factories.get_container(etcd_container_name, image_name=etcd_container_image_name, container_run_kwargs={'environment': container_environment, 'ports': {'2379/tcp': etcd_static_port}}, pull_before_start=True, skip_on_pull_failure=True, skip_if_docker_client_not_connectable=True)
    container.container_start_check(confirm_container_started, container)
    with container.started() as factory:
        yield factory

@pytest.fixture(scope='module')
def etcd_port(etcd_container):
    if False:
        print('Hello World!')
    return etcd_container.get_host_port_binding(2379, protocol='tcp', ipv6=False)

@pytest.fixture(scope='module')
def profile_name():
    if False:
        i = 10
        return i + 15
    return 'etcd_util_profile'

@pytest.fixture(scope='module')
def etcd_profile(profile_name, etcd_port, etcd_version):
    if False:
        for i in range(10):
            print('nop')
    profile = {profile_name: {'etcd.host': '127.0.0.1', 'etcd.port': etcd_port, 'etcd.require_v2': etcd_version in (EtcdVersion.v2, EtcdVersion.v3_v2_mode)}}
    return profile