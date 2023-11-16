import json
import pathlib
import tempfile
import time
import pytest
from tests.conftest import CODE_DIR
docker = pytest.importorskip('docker')
pytestmark = [pytest.mark.core_test]

def json_output_to_dict(output):
    if False:
        return 10
    '\n    Convert ``salt ... --out=json`` Syndic return to a dictionary. Since the\n    --out=json will return several JSON outputs, e.g. {...}\\n{...}, we have to\n    parse that output individually.\n    '
    output = output or ''
    results = {}
    for line in (_ for _ in output.replace('\n}', '\n}\x1f').split('\x1f') if _.strip()):
        data = json.loads(line)
        if isinstance(data, dict):
            for minion in data:
                if minion not in ('syndic_a', 'syndic_b'):
                    results[minion] = data[minion]
    return results

def accept_keys(container, required_minions):
    if False:
        i = 10
        return i + 15
    failure_time = time.time() + 20
    while time.time() < failure_time:
        container.run('salt-key -Ay')
        res = container.run('salt-key -L --out=json')
        if isinstance(res.data, dict) and set(res.data.get('minions')) == required_minions:
            break
    else:
        pytest.skip(f'{container} unable to accept keys for {required_minions}')

@pytest.fixture(scope='module')
def syndic_network():
    if False:
        return 10
    try:
        client = docker.from_env()
    except docker.errors.DockerException as e:
        pytest.skip(f'Docker failed with error {e}')
    pool = docker.types.IPAMPool(subnet='172.27.13.0/24', gateway='172.27.13.1')
    ipam_config = docker.types.IPAMConfig(pool_configs=[pool])
    network = None
    try:
        network = client.networks.create(name='syndic_test_net', ipam=ipam_config)
        yield network.name
    finally:
        if network is not None:
            network.remove()

@pytest.fixture(scope='module')
def source_path():
    if False:
        print('Hello World!')
    return str(CODE_DIR / 'salt')

@pytest.fixture(scope='module')
def container_image_name():
    if False:
        i = 10
        return i + 15
    return 'ghcr.io/saltstack/salt-ci-containers/salt:3005'

@pytest.fixture(scope='module')
def container_python_version():
    if False:
        return 10
    return '3.7'

@pytest.fixture(scope='module')
def config(source_path):
    if False:
        i = 10
        return i + 15
    with tempfile.TemporaryDirectory() as tmp_path:
        tmp_path = pathlib.Path(tmp_path)
        master_dir = tmp_path / 'master'
        minion_dir = tmp_path / 'minion'
        syndic_a_dir = tmp_path / 'syndic_a'
        syndic_b_dir = tmp_path / 'syndic_b'
        minion_a1_dir = tmp_path / 'minion_a1'
        minion_a2_dir = tmp_path / 'minion_a2'
        minion_b1_dir = tmp_path / 'minion_b1'
        minion_b2_dir = tmp_path / 'minion_b2'
        for dir_ in (master_dir, minion_dir, syndic_a_dir, syndic_b_dir, minion_a1_dir, minion_a2_dir, minion_b1_dir, minion_b2_dir):
            dir_.mkdir(parents=True, exist_ok=True)
            (dir_ / 'master.d').mkdir(exist_ok=True)
            (dir_ / 'minion.d').mkdir(exist_ok=True)
            (dir_ / 'pki').mkdir(exist_ok=True)
        (master_dir / 'master.d').mkdir(exist_ok=True)
        master_config_path = master_dir / 'master'
        master_config_path.write_text('\nauth.pam.python: /usr/local/bin/python3\norder_masters: True\n\npublisher_acl:\n  bob:\n    - \'*1\':\n      - test.*\n      - file.touch\n\nexternal_auth:\n  pam:\n    bob:\n      - \'*1\':\n        - test.*\n        - file.touch\n\nnodegroups:\n  second_string: "minion_*2"\n  b_string: "minion_b*"\n\n        ')
        minion_config_path = minion_dir / 'minion'
        minion_config_path.write_text('id: minion\nmaster: master')
        syndic_a_minion_config_path = syndic_a_dir / 'minion'
        syndic_a_minion_config_path.write_text('id: syndic_a\nmaster: master')
        syndic_a_master_config_path = syndic_a_dir / 'master'
        syndic_a_master_config_path.write_text("\nauth.pam.python: /usr/local/bin/python3\nsyndic_master: master\npublisher_acl:\n  bob:\n    - '*1':\n      - test.*\n      - file.touch\n\nexternal_auth:\n  pam:\n    bob:\n      - '*1':\n        - test.*\n        - file.touch\n        ")
        minion_a1_config_path = minion_a1_dir / 'minion'
        minion_a1_config_path.write_text('id: minion_a1\nmaster: syndic_a')
        minion_a2_config_path = minion_a2_dir / 'minion'
        minion_a2_config_path.write_text('id: minion_a2\nmaster: syndic_a')
        syndic_b_minion_config_path = syndic_b_dir / 'minion'
        syndic_b_minion_config_path.write_text('id: syndic_b\nmaster: master')
        syndic_b_master_config_path = syndic_b_dir / 'master'
        syndic_b_master_config_path.write_text('syndic_master: master')
        minion_b1_config_path = minion_b1_dir / 'minion'
        minion_b1_config_path.write_text('id: minion_b1\nmaster: syndic_b')
        minion_b2_config_path = minion_b2_dir / 'minion'
        minion_b2_config_path.write_text('id: minion_b2\nmaster: syndic_b')
        yield {'minion_dir': minion_dir, 'master_dir': master_dir, 'syndic_a_dir': syndic_a_dir, 'syndic_b_dir': syndic_b_dir, 'minion_a1_dir': minion_a1_dir, 'minion_a2_dir': minion_a2_dir, 'minion_b1_dir': minion_b1_dir, 'minion_b2_dir': minion_b2_dir}

@pytest.fixture(scope='module')
def docker_master(salt_factories, syndic_network, config, source_path, container_image_name, container_python_version):
    if False:
        print('Hello World!')
    config_dir = str(config['master_dir'])
    container = salt_factories.get_container('master', image_name=container_image_name, container_run_kwargs={'entrypoint': 'python -m http.server', 'network': syndic_network, 'volumes': {config_dir: {'bind': '/etc/salt', 'mode': 'z'}, source_path: {'bind': f'/usr/local/lib/python{container_python_version}/site-packages/salt/', 'mode': 'z'}}}, pull_before_start=True, skip_on_pull_failure=True, skip_if_docker_client_not_connectable=True)
    with container.started() as factory:
        for user in ('bob', 'fnord'):
            ret = container.run(f'adduser {user}')
            assert ret.returncode == 0
            ret = container.run(f'passwd -d {user}')
            assert ret.returncode == 0
        yield factory

@pytest.fixture(scope='module')
def docker_minion(salt_factories, syndic_network, config, source_path, container_image_name, container_python_version):
    if False:
        for i in range(10):
            print('nop')
    config_dir = str(config['minion_dir'])
    container = salt_factories.get_container('minion', image_name=container_image_name, container_run_kwargs={'entrypoint': 'python -m http.server', 'network': syndic_network, 'volumes': {config_dir: {'bind': '/etc/salt', 'mode': 'z'}, source_path: {'bind': f'/usr/local/lib/python{container_python_version}/site-packages/salt/', 'mode': 'z'}}}, pull_before_start=True, skip_on_pull_failure=True, skip_if_docker_client_not_connectable=True)
    with container.started() as factory:
        yield factory

@pytest.fixture(scope='module')
def docker_syndic_a(salt_factories, config, syndic_network, source_path, container_image_name, container_python_version):
    if False:
        i = 10
        return i + 15
    config_dir = str(config['syndic_a_dir'])
    container = salt_factories.get_container('syndic_a', image_name=container_image_name, container_run_kwargs={'entrypoint': 'python -m http.server', 'network': syndic_network, 'volumes': {config_dir: {'bind': '/etc/salt', 'mode': 'z'}, source_path: {'bind': f'/usr/local/lib/python{container_python_version}/site-packages/salt/', 'mode': 'z'}}}, pull_before_start=True, skip_on_pull_failure=True, skip_if_docker_client_not_connectable=True)
    with container.started() as factory:
        yield factory

@pytest.fixture(scope='module')
def docker_syndic_b(salt_factories, config, syndic_network, source_path, container_image_name, container_python_version):
    if False:
        print('Hello World!')
    config_dir = str(config['syndic_b_dir'])
    container = salt_factories.get_container('syndic_b', image_name=container_image_name, container_run_kwargs={'entrypoint': 'python -m http.server', 'network': syndic_network, 'volumes': {config_dir: {'bind': '/etc/salt', 'mode': 'z'}, source_path: {'bind': f'/usr/local/lib/python{container_python_version}/site-packages/salt/', 'mode': 'z'}}}, pull_before_start=True, skip_on_pull_failure=True, skip_if_docker_client_not_connectable=True)
    with container.started() as factory:
        yield factory

@pytest.fixture(scope='module')
def docker_minion_a1(salt_factories, config, syndic_network, source_path, container_image_name, container_python_version):
    if False:
        while True:
            i = 10
    config_dir = str(config['minion_a1_dir'])
    container = salt_factories.get_container('minion_a1', image_name=container_image_name, container_run_kwargs={'network': syndic_network, 'entrypoint': 'python -m http.server', 'volumes': {config_dir: {'bind': '/etc/salt', 'mode': 'z'}, source_path: {'bind': f'/usr/local/lib/python{container_python_version}/site-packages/salt/', 'mode': 'z'}}}, pull_before_start=True, skip_on_pull_failure=True, skip_if_docker_client_not_connectable=True)
    with container.started() as factory:
        yield factory

@pytest.fixture(scope='module')
def docker_minion_a2(salt_factories, config, syndic_network, source_path, container_image_name, container_python_version):
    if False:
        while True:
            i = 10
    config_dir = str(config['minion_a2_dir'])
    container = salt_factories.get_container('minion_a2', image_name=container_image_name, container_run_kwargs={'network': syndic_network, 'entrypoint': 'python -m http.server', 'volumes': {config_dir: {'bind': '/etc/salt', 'mode': 'z'}, source_path: {'bind': f'/usr/local/lib/python{container_python_version}/site-packages/salt/', 'mode': 'z'}}}, pull_before_start=True, skip_on_pull_failure=True, skip_if_docker_client_not_connectable=True)
    with container.started() as factory:
        yield factory

@pytest.fixture(scope='module')
def docker_minion_b1(salt_factories, config, syndic_network, source_path, container_image_name, container_python_version):
    if False:
        for i in range(10):
            print('nop')
    config_dir = str(config['minion_b1_dir'])
    container = salt_factories.get_container('minion_b1', image_name=container_image_name, container_run_kwargs={'network': syndic_network, 'entrypoint': 'python -m http.server', 'volumes': {config_dir: {'bind': '/etc/salt', 'mode': 'z'}, source_path: {'bind': f'/usr/local/lib/python{container_python_version}/site-packages/salt/', 'mode': 'z'}}}, pull_before_start=True, skip_on_pull_failure=True, skip_if_docker_client_not_connectable=True)
    with container.started() as factory:
        yield factory

@pytest.fixture(scope='module')
def docker_minion_b2(salt_factories, config, syndic_network, source_path, container_image_name, container_python_version):
    if False:
        i = 10
        return i + 15
    config_dir = str(config['minion_b2_dir'])
    container = salt_factories.get_container('minion_b2', image_name=container_image_name, container_run_kwargs={'network': syndic_network, 'entrypoint': 'python -m http.server', 'volumes': {config_dir: {'bind': '/etc/salt', 'mode': 'z'}, source_path: {'bind': f'/usr/local/lib/python{container_python_version}/site-packages/salt/', 'mode': 'z'}}}, pull_before_start=True, skip_on_pull_failure=True, skip_if_docker_client_not_connectable=True)
    with container.started() as factory:
        yield factory

@pytest.fixture(scope='module', autouse=True)
def all_the_docker(docker_master, docker_minion, docker_syndic_a, docker_syndic_b, docker_minion_a1, docker_minion_a2, docker_minion_b1, docker_minion_b2):
    if False:
        for i in range(10):
            print('nop')
    try:
        for s in (docker_master, docker_syndic_a, docker_syndic_b, docker_minion_a1, docker_minion_a2, docker_minion_b1, docker_minion_b2, docker_minion):
            s.run('python3 -m pip install looseversion packaging')
        for s in (docker_master, docker_syndic_a, docker_syndic_b):
            s.run('salt-master -d -ldebug')
        for s in (docker_minion_a1, docker_minion_a2, docker_minion_b1, docker_minion_b2, docker_minion):
            s.run('salt-minion -d')
        for s in (docker_syndic_a, docker_syndic_b):
            s.run('salt-syndic -d')
        failure_time = time.time() + 20
        accept_keys(container=docker_master, required_minions={'minion', 'syndic_a', 'syndic_b'})
        accept_keys(container=docker_syndic_a, required_minions={'minion_a1', 'minion_a2'})
        accept_keys(container=docker_syndic_b, required_minions={'minion_b1', 'minion_b2'})
        for tries in range(30):
            res = docker_master.run('salt \\* test.ping -t20 --out=json')
            results = json_output_to_dict(res.stdout)
            if set(results).issuperset(['minion', 'minion_a1', 'minion_a2', 'minion_b1', 'minion_b2']):
                break
        else:
            pytest.skip(f'Missing some minions: {sorted(results)}')
        yield
    finally:
        for container in (docker_minion, docker_syndic_a, docker_syndic_b, docker_minion_a1, docker_minion_a2, docker_minion_b1, docker_minion_b2):
            try:
                container.run('rm -rfv /etc/salt/')
            except docker.errors.APIError as e:
                print(f'Docker failed removing /etc/salt: {e}')

@pytest.fixture(params=[('*', ['minion', 'minion_a1', 'minion_a2', 'minion_b1', 'minion_b2']), ('minion', ['minion']), ('minion_*', ['minion_a1', 'minion_a2', 'minion_b1', 'minion_b2']), ('minion_a*', ['minion_a1', 'minion_a2']), ('minion_b*', ['minion_b1', 'minion_b2']), ('*1', ['minion_a1', 'minion_b1']), ('*2', ['minion_a2', 'minion_b2'])])
def all_the_minions(request):
    if False:
        print('Hello World!')
    yield request.param

@pytest.fixture(params=[('minion_a1', ['minion_a1']), ('minion_b1', ['minion_b1']), ('*1', ['minion_a1', 'minion_b1']), ('minion*1', ['minion_a1', 'minion_b1'])])
def eauth_valid_minions(request):
    if False:
        while True:
            i = 10
    yield request.param

@pytest.fixture(params=['*', 'minion', 'minion_a2', 'minion_b2', 'syndic_a', 'syndic_b', '*2', 'minion*', 'minion_a*', 'minion_b*'])
def eauth_blocked_minions(request):
    if False:
        i = 10
        return i + 15
    yield request.param

@pytest.fixture
def docker_minions(docker_minion, docker_minion_a1, docker_minion_a2, docker_minion_b1, docker_minion_b2):
    if False:
        while True:
            i = 10
    yield [docker_minion, docker_minion_a1, docker_minion_a2, docker_minion_b1, docker_minion_b2]

@pytest.fixture(params=['test.arg good_argument', 'test.arg bad_news', 'test.arg not_allowed', 'test.echo very_not_good', "cmd.run 'touch /tmp/fun.txt'", 'file.touch /tmp/more_fun.txt', 'test.arg_repr this_is_whatever', 'test.arg_repr more whatever', 'test.arg_repr cool guy'])
def all_the_commands(request):
    if False:
        while True:
            i = 10
    yield request.param

@pytest.fixture(params=['test.arg', 'test.echo'])
def eauth_valid_commands(request):
    if False:
        print('Hello World!')
    yield request.param

@pytest.fixture(params=['cmd.run', 'file.manage_file', 'test.arg_repr'])
def eauth_invalid_commands(request):
    if False:
        return 10
    yield request.param

@pytest.fixture(params=['good_argument', 'good_things', 'good_super_awesome_stuff'])
def eauth_valid_arguments(request):
    if False:
        while True:
            i = 10
    yield request.param

@pytest.fixture(params=['bad_news', 'not_allowed', 'very_not_good'])
def eauth_invalid_arguments(request):
    if False:
        for i in range(10):
            print('nop')
    yield request.param

@pytest.fixture(params=['G@id:minion_a1 and minion_b*', 'E@minion_[^b]1 and minion_b2', 'P@id:minion_[^b]. and minion'])
def invalid_comprehensive_minion_targeting(request):
    if False:
        for i in range(10):
            print('nop')
    yield request.param

@pytest.fixture(params=[('G@id:minion or minion_a1 or E@minion_[^b]2 or L@minion_b1,minion_b2', ['minion', 'minion_a1', 'minion_a2', 'minion_b1', 'minion_b2']), ('minion or E@minion_a[12] or N@b_string', ['minion', 'minion_a1', 'minion_a2', 'minion_b1', 'minion_b2']), ('L@minion,minion_a1 or N@second_string or N@b_string', ['minion', 'minion_a1', 'minion_a2', 'minion_b1', 'minion_b2'])])
def comprehensive_minion_targeting(request):
    if False:
        while True:
            i = 10
    yield request.param

@pytest.fixture(params=[('G@id:minion_a1 and minion_b1', ['minion_a1', 'minion_b1']), ('E@minion_[^b]1', ['minion_a1']), ('P@id:minion_[^b].', ['minion_a1', 'minion_a2']), ('L@minion_a1,minion_a2,minion_b1 not minion_*2', ['minion_a1', 'minion_a2', 'minion_b1'])])
def valid_comprehensive_minion_targeting(request):
    if False:
        for i in range(10):
            print('nop')
    yield request.param

@pytest.fixture(params=[('E@minion_[^b]1', ['minion_a1']), ('P@id:minion_[^a]1', ['minion_b1']), ('L@minion_a1,minion_b1 not minion_*2', ['minion_a1', 'minion_b1'])])
def valid_eauth_comprehensive_minion_targeting(request):
    if False:
        print('Hello World!')
    yield request.param

def test_root_user_should_be_able_to_call_any_and_all_minions_with_any_and_all_commands(all_the_minions, all_the_commands, docker_master):
    if False:
        i = 10
        return i + 15
    (target, expected_minions) = all_the_minions
    res = docker_master.run(f'salt {target} {all_the_commands} -t 20 --out=json')
    if 'jid does not exist' in (res.stderr or ''):
        res = docker_master.run(f'salt {target} {all_the_commands} -t 20 --out=json')
    results = json_output_to_dict(res.stdout)
    assert sorted(results) == expected_minions, res.stdout

def test_eauth_user_should_be_able_to_target_valid_minions_with_valid_command(eauth_valid_minions, eauth_valid_commands, eauth_valid_arguments, docker_master):
    if False:
        return 10
    (target, expected_minions) = eauth_valid_minions
    res = docker_master.run(f"salt -a pam --username bob --password '' {target} {eauth_valid_commands} {eauth_valid_arguments} -t 20 --out=json")
    results = json_output_to_dict(res.stdout)
    assert sorted(results) == expected_minions, res.stdout

def test_eauth_user_should_not_be_able_to_target_invalid_minions(eauth_blocked_minions, docker_master, docker_minions):
    if False:
        print('Hello World!')
    res = docker_master.run(f"salt -a pam --username bob --password '' {eauth_blocked_minions} file.touch /tmp/bad_bad_file.txt -t 20 --out=json")
    assert 'Authorization error occurred.' == res.data or res.data is None
    for minion in docker_minions:
        res = minion.run('test -f /tmp/bad_bad_file.txt')
        file_exists = res.returncode == 0
        assert not file_exists

@pytest.mark.skip(reason='Not sure about blocklist')
def test_eauth_user_should_not_be_able_to_target_valid_minions_with_invalid_commands(eauth_valid_minions, eauth_invalid_commands, docker_master):
    if False:
        i = 10
        return i + 15
    (tgt, _) = eauth_valid_minions
    res = docker_master.run(f"salt -a pam --username bob --password '' {tgt} {eauth_invalid_commands} -t 20 --out=json")
    results = json_output_to_dict(res.stdout)
    assert 'Authorization error occurred' in res.stdout
    assert sorted(results) == []

@pytest.mark.skip(reason='Not sure about blocklist')
def test_eauth_user_should_not_be_able_to_target_valid_minions_with_valid_commands_and_invalid_arguments(eauth_valid_minions, eauth_valid_commands, eauth_invalid_arguments, docker_master):
    if False:
        i = 10
        return i + 15
    (tgt, _) = eauth_valid_minions
    res = docker_master.run(f"salt -a pam --username bob --password '' -C '{tgt}' {eauth_valid_commands} {eauth_invalid_arguments} -t 20 --out=json")
    results = json_output_to_dict(res.stdout)
    assert 'Authorization error occurred' in res.stdout
    assert sorted(results) == []

def test_invalid_eauth_user_should_not_be_able_to_do_anything(eauth_valid_minions, eauth_valid_commands, eauth_valid_arguments, docker_master):
    if False:
        for i in range(10):
            print('nop')
    (tgt, _) = eauth_valid_minions
    res = docker_master.run(f"salt -a pam --username badguy --password '' -C '{tgt}' {eauth_valid_commands} {eauth_valid_arguments} -t 20 --out=json")
    results = json_output_to_dict(res.stdout)
    assert sorted(results) == []

def test_root_should_be_able_to_use_comprehensive_targeting(comprehensive_minion_targeting, docker_master):
    if False:
        i = 10
        return i + 15
    (tgt, expected_minions) = comprehensive_minion_targeting
    res = docker_master.run(f"salt -C '{tgt}' test.version -t 20 --out=json")
    results = json_output_to_dict(res.stdout)
    assert sorted(results) == expected_minions

def test_eauth_user_should_be_able_to_target_valid_minions_with_valid_commands_comprehensively(valid_eauth_comprehensive_minion_targeting, docker_master):
    if False:
        i = 10
        return i + 15
    (tgt, expected_minions) = valid_eauth_comprehensive_minion_targeting
    res = docker_master.run(f"salt -a pam --username bob --password '' -C '{tgt}' test.version -t 20 --out=json")
    results = json_output_to_dict(res.stdout)
    assert sorted(results) == expected_minions

def test_eauth_user_with_invalid_comprehensive_targeting_should_auth_failure(invalid_comprehensive_minion_targeting, docker_master):
    if False:
        while True:
            i = 10
    res = docker_master.run(f"salt -a pam --username fnord --password '' -C '{invalid_comprehensive_minion_targeting}' test.version -t 20 --out=json")
    results = json_output_to_dict(res.stdout)
    assert 'Authorization error occurred' in res.stdout
    assert sorted(results) == []