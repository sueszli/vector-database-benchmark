import pytest
pytestmark = [pytest.mark.skip_on_windows(reason='Not supported on Windows'), pytest.mark.skip_if_binaries_missing('ansible', 'ansible-doc', 'ansible-playbook', check_all=True, reason='ansible is not installed'), pytest.mark.slow_test]

@pytest.fixture
def ansible(modules):
    if False:
        i = 10
        return i + 15
    return modules.ansible

def test_short_alias(modules, ansible):
    if False:
        print('Hello World!')
    '\n    Test that the ansible functions are actually loaded and we can target using the short alias.\n    '
    ret = ansible.ping()
    assert ret == {'ping': 'pong'}
    ansible_ping_func = None
    if 'ansible.system.ping' in modules:
        ansible_ping_func = getattr(modules.ansible, 'system.ping')
        assert 'ansible.ansible.system.ping' not in modules
    elif 'ansible.builtin.ping' in modules:
        ansible_ping_func = getattr(modules.ansible, 'builtin.ping')
        assert 'ansible.ansible.builtin.ping' not in modules
    if ansible_ping_func:
        ret = ansible_ping_func()
        assert ret == {'ping': 'pong'}

def test_passing_data_to_ansible_modules(ansible):
    if False:
        return 10
    '\n    Test that the ansible functions are actually loaded\n    '
    expected = 'foobar'
    ret = ansible.ping(data=expected)
    assert ret == {'ping': expected}