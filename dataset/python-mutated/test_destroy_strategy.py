"""Testinfra tests."""
import os
import testinfra.utils.ansible_runner
testinfra_hosts = testinfra.utils.ansible_runner.AnsibleRunner(os.environ['MOLECULE_INVENTORY_FILE']).get_hosts('all')

def test_hostname(host):
    if False:
        return 10
    'Validate hostname.'
    assert host.check_output('hostname -s') == 'instance'

def test_etc_molecule_directory(host):
    if False:
        i = 10
        return i + 15
    'Validate molecule directory.'
    f = host.file('/etc/molecule')
    assert f.is_directory
    assert f.user == 'root'
    assert f.group == 'root'
    assert f.mode == 493

def test_etc_molecule_ansible_hostname_file(host):
    if False:
        i = 10
        return i + 15
    'Validate molecule instance file.'
    f = host.file('/etc/molecule/instance')
    assert f.is_file
    assert f.user == 'root'
    assert f.group == 'root'
    assert f.mode == 420