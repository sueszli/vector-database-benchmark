"""Testinfra tests."""
import os
import testinfra.utils.ansible_runner
testinfra_hosts = testinfra.utils.ansible_runner.AnsibleRunner(os.environ['MOLECULE_INVENTORY_FILE']).get_hosts('all')

def test_side_effect_removed_file(host):
    if False:
        for i in range(10):
            print('nop')
    'Validate that file was removed.'
    assert not host.file('/tmp/testfile').exists