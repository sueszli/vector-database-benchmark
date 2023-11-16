import os
import pytest
import yaml
REPO_ROOT = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir))
ANSIBLE_BASE = os.path.join(REPO_ROOT, 'install_files', 'ansible-base')

def find_ansible_playbooks():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test helper to generate list of filepaths for SecureDrop\n    Ansible playbooks. All files will be validated to contain the\n    max_fail option.\n    '
    playbooks = []
    for f in os.listdir(ANSIBLE_BASE):
        if f.endswith('.yml'):
            if f not in ['prod-specific.yml', 'build-deb-pkgs.yml']:
                playbooks.append(os.path.join(ANSIBLE_BASE, f))
    assert len(playbooks) > 0
    return playbooks

@pytest.mark.parametrize('playbook', find_ansible_playbooks())
def test_max_fail_percentage(host, playbook):
    if False:
        return 10
    '\n    All SecureDrop playbooks should set `max_fail_percentage` to "0"\n    on each and every play. Doing so ensures that an error on a single\n    host constitutes a play failure.\n\n    In conjunction with the `any_errors_fatal` option, tested separately,\n    this will achieve a "fail fast" behavior from Ansible.\n\n    There\'s no ansible.cfg option to set for max_fail_percentage, which would\n    allow for a single DRY update that would apply automatically to all\n    invocations of `ansible-playbook`. Therefore this test, which will\n    search for the line present in all playbooks.\n\n    Technically it\'s only necessary that plays targeting multiple hosts use\n    the parameter, but we\'ll play it safe and require it everywhere,\n    to avoid mistakes down the road.\n    '
    with open(playbook) as f:
        playbook_yaml = yaml.safe_load(f)
        for play in playbook_yaml:
            assert 'max_fail_percentage' in play
            assert play['max_fail_percentage'] == 0

@pytest.mark.parametrize('playbook', find_ansible_playbooks())
def test_any_errors_fatal(host, playbook):
    if False:
        print('Hello World!')
    '\n    All SecureDrop playbooks should set `any_errors_fatal` to "yes"\n    on each and every play. In conjunction with `max_fail_percentage` set\n    to "0", doing so ensures that any errors will cause an immediate failure\n    on the playbook.\n    '
    with open(playbook) as f:
        playbook_yaml = yaml.safe_load(f)
        for play in playbook_yaml:
            assert 'any_errors_fatal' in play
            assert play['any_errors_fatal']

@pytest.mark.parametrize('playbook', find_ansible_playbooks())
def test_locale(host, playbook):
    if False:
        while True:
            i = 10
    '\n    The securedrop-prod and securedrop-staging playbooks should\n    control the locale in the host environment by setting LC_ALL=C.\n    '
    with open(os.path.join(ANSIBLE_BASE, playbook)) as f:
        playbook_yaml = yaml.safe_load(f)
        for play in playbook_yaml:
            assert 'environment' in play
            assert play['environment']['LC_ALL'] == 'C'