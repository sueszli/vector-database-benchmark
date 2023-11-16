import pytest
from thefuck.rules.apt_upgrade import get_new_command, match
from thefuck.types import Command
match_output = '\nListing... Done\nheroku/stable 6.15.2-1 amd64 [upgradable from: 6.14.43-1]\nresolvconf/zesty-updates,zesty-updates 1.79ubuntu4.1 all [upgradable from: 1.79ubuntu4]\nsquashfs-tools/zesty-updates 1:4.3-3ubuntu2.17.04.1 amd64 [upgradable from: 1:4.3-3ubuntu2]\nunattended-upgrades/zesty-updates,zesty-updates 0.93.1ubuntu2.4 all [upgradable from: 0.93.1ubuntu2.3]\n'
no_match_output = '\nListing... Done\n'

def test_match():
    if False:
        while True:
            i = 10
    assert match(Command('apt list --upgradable', match_output))
    assert match(Command('sudo apt list --upgradable', match_output))

@pytest.mark.parametrize('command', [Command('apt list --upgradable', no_match_output), Command('sudo apt list --upgradable', no_match_output)])
def test_not_match(command):
    if False:
        print('Hello World!')
    assert not match(command)

def test_get_new_command():
    if False:
        for i in range(10):
            print('nop')
    new_command = get_new_command(Command('apt list --upgradable', match_output))
    assert new_command == 'apt upgrade'
    new_command = get_new_command(Command('sudo apt list --upgradable', match_output))
    assert new_command == 'sudo apt upgrade'