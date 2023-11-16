"""
Tests for logrotate config
"""
import pathlib
import packaging.version
import pytest
pytestmark = [pytest.mark.skip_unless_on_linux]

@pytest.fixture
def logrotate_config_file(grains):
    if False:
        return 10
    '\n    Fixture for logrotate config file path\n    '
    if grains['os_family'] == 'RedHat':
        return pathlib.Path('/etc/logrotate.d', 'salt')
    elif grains['os_family'] == 'Debian':
        return pathlib.Path('/etc/logrotate.d', 'salt-common')

def test_logrotate_config(logrotate_config_file):
    if False:
        while True:
            i = 10
    '\n    Test that logrotate config has been installed in correctly\n    '
    assert logrotate_config_file.is_file()
    assert logrotate_config_file.owner() == 'root'
    assert logrotate_config_file.group() == 'root'

def test_issue_65231_etc_logrotate_salt_dir_removed(install_salt):
    if False:
        print('Hello World!')
    '\n    Test that /etc/logrotate.d/salt is not a directory\n    '
    if install_salt.prev_version and packaging.version.parse(install_salt.prev_version) <= packaging.version.parse('3006.4'):
        pytest.skip('Testing a downgrade to 3006.4, do not run')
    path = pathlib.Path('/etc/logrotate.d/salt')
    if path.exists():
        assert path.is_dir() is False