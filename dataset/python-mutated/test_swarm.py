import pytest
import salt.utils.versions
pytestmark = [pytest.mark.slow_test, pytest.mark.skip_if_binaries_missing('dockerd')]
pytest.importorskip('docker')

@pytest.fixture
def docker_version(shell, grains):
    if False:
        return 10
    ret = shell.run('docker', '--version')
    assert ret.returncode == 0
    return salt.utils.versions.Version(ret.stdout.split(',')[0].split()[-1].strip())

@pytest.fixture
def swarm(modules):
    if False:
        i = 10
        return i + 15
    try:
        yield modules.swarm
    finally:
        modules.swarm.leave_swarm(force=True)

def test_swarm_init(swarm, docker_version, grains):
    if False:
        while True:
            i = 10
    '\n    check that swarm.swarm_init works when a swarm exists\n    '
    ret = swarm.swarm_init('127.0.0.1', '0.0.0.0', False)
    if 'Error' in ret:
        if ret['Error'] == '--live-restore daemon configuration is incompatible with swarm mode':
            if grains['os'] == 'Fedora' and int(grains['osmajorrelease']) >= 37:
                pytest.skip(f"The 'swarm' module needs to be updated to support the system docker {docker_version} package")
    assert 'Error' not in ret
    for key in ('Comment', 'Tokens'):
        assert key in ret
    assert ret['Comment'].startswith('Docker swarm has been initialized on ')
    ret = swarm.swarm_init('127.0.0.1', '0.0.0.0', False)
    expected = {'Comment': 'This node is already part of a swarm. Use "docker swarm leave" to leave this swarm and join another one.', 'result': False}
    assert ret == expected