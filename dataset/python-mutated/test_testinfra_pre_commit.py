"""Testinfra tests."""

def test_ansible_hostname(host):
    if False:
        i = 10
        return i + 15
    'Validate hostname.'
    f = host.file('/tmp/molecule/instance-1')
    assert not f.exists