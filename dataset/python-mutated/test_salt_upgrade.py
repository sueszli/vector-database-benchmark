import packaging.version
import pytest

def test_salt_upgrade(salt_call_cli, install_salt):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test an upgrade of Salt.\n    '
    if not install_salt.upgrade:
        pytest.skip('Not testing an upgrade, do not run')
    if install_salt.relenv:
        original_py_version = install_salt.package_python_version()
    ret = salt_call_cli.run('test.version')
    assert ret.returncode == 0
    assert packaging.version.parse(ret.data) < packaging.version.parse(install_salt.artifact_version)
    dep = 'PyGithub==1.56.0'
    install = salt_call_cli.run('--local', 'pip.install', dep)
    assert install.returncode == 0
    repo = 'https://github.com/saltstack/salt.git'
    use_lib = salt_call_cli.run('--local', 'github.get_repo_info', repo)
    assert 'Authentication information could' in use_lib.stderr
    install_salt.install(upgrade=True)
    ret = salt_call_cli.run('test.version')
    assert ret.returncode == 0
    assert packaging.version.parse(ret.data) == packaging.version.parse(install_salt.artifact_version)
    if install_salt.relenv:
        new_py_version = install_salt.package_python_version()
        if new_py_version == original_py_version:
            use_lib = salt_call_cli.run('--local', 'github.get_repo_info', repo)