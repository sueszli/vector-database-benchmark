import pytest
from thefuck.rules.apt_list_upgradable import get_new_command, match
from thefuck.types import Command
full_english_output = "\nHit:1 http://us.archive.ubuntu.com/ubuntu zesty InRelease\nHit:2 http://us.archive.ubuntu.com/ubuntu zesty-updates InRelease\nGet:3 http://us.archive.ubuntu.com/ubuntu zesty-backports InRelease [89.2 kB]\nHit:4 http://security.ubuntu.com/ubuntu zesty-security InRelease\nHit:5 http://ppa.launchpad.net/ubuntu-mozilla-daily/ppa/ubuntu zesty InRelease\nHit:6 https://download.docker.com/linux/ubuntu zesty InRelease\nHit:7 https://cli-assets.heroku.com/branches/stable/apt ./ InRelease\nFetched 89.2 kB in 0s (122 kB/s)\nReading package lists... Done\nBuilding dependency tree\nReading state information... Done\n8 packages can be upgraded. Run 'apt list --upgradable' to see them.\n"
match_output = [full_english_output, 'Führen Sie »apt list --upgradable« aus, um sie anzuzeigen.']
no_match_output = '\nHit:1 http://us.archive.ubuntu.com/ubuntu zesty InRelease\nGet:2 http://us.archive.ubuntu.com/ubuntu zesty-updates InRelease [89.2 kB]\nGet:3 http://us.archive.ubuntu.com/ubuntu zesty-backports InRelease [89.2 kB]\nGet:4 http://security.ubuntu.com/ubuntu zesty-security InRelease [89.2 kB]\nHit:5 https://cli-assets.heroku.com/branches/stable/apt ./ InRelease\nHit:6 http://ppa.launchpad.net/ubuntu-mozilla-daily/ppa/ubuntu zesty InRelease\nHit:7 https://download.docker.com/linux/ubuntu zesty InRelease\nGet:8 http://us.archive.ubuntu.com/ubuntu zesty-updates/main i386 Packages [232 kB]\nGet:9 http://us.archive.ubuntu.com/ubuntu zesty-updates/main amd64 Packages [235 kB]\nGet:10 http://us.archive.ubuntu.com/ubuntu zesty-updates/main amd64 DEP-11 Metadata [55.2 kB]\nGet:11 http://us.archive.ubuntu.com/ubuntu zesty-updates/main DEP-11 64x64 Icons [32.3 kB]\nGet:12 http://us.archive.ubuntu.com/ubuntu zesty-updates/universe amd64 Packages [156 kB]\nGet:13 http://us.archive.ubuntu.com/ubuntu zesty-updates/universe i386 Packages [156 kB]\nGet:14 http://us.archive.ubuntu.com/ubuntu zesty-updates/universe amd64 DEP-11 Metadata [175 kB]\nGet:15 http://us.archive.ubuntu.com/ubuntu zesty-updates/universe DEP-11 64x64 Icons [253 kB]\nGet:16 http://us.archive.ubuntu.com/ubuntu zesty-updates/multiverse amd64 DEP-11 Metadata [5,840 B]\nGet:17 http://us.archive.ubuntu.com/ubuntu zesty-backports/universe amd64 DEP-11 Metadata [4,588 B]\nGet:18 http://security.ubuntu.com/ubuntu zesty-security/main amd64 DEP-11 Metadata [12.7 kB]\nGet:19 http://security.ubuntu.com/ubuntu zesty-security/main DEP-11 64x64 Icons [17.6 kB]\nGet:20 http://security.ubuntu.com/ubuntu zesty-security/universe amd64 DEP-11 Metadata [21.6 kB]\nGet:21 http://security.ubuntu.com/ubuntu zesty-security/universe DEP-11 64x64 Icons [47.7 kB]\nGet:22 http://security.ubuntu.com/ubuntu zesty-security/multiverse amd64 DEP-11 Metadata [208 B]\nFetched 1,673 kB in 0s (1,716 kB/s)\nReading package lists... Done\nBuilding dependency tree\nReading state information... Done\nAll packages are up to date.\n'

@pytest.mark.parametrize('output', match_output)
def test_match(output):
    if False:
        i = 10
        return i + 15
    assert match(Command('sudo apt update', output))

@pytest.mark.parametrize('command', [Command('apt-cache search foo', ''), Command('aptitude search foo', ''), Command('apt search foo', ''), Command('apt-get install foo', ''), Command('apt-get source foo', ''), Command('apt-get clean', ''), Command('apt-get remove', ''), Command('apt-get update', ''), Command('sudo apt update', no_match_output)])
def test_not_match(command):
    if False:
        return 10
    assert not match(command)

@pytest.mark.parametrize('output', match_output)
def test_get_new_command(output):
    if False:
        print('Hello World!')
    new_command = get_new_command(Command('sudo apt update', output))
    assert new_command == 'sudo apt list --upgradable'
    new_command = get_new_command(Command('apt update', output))
    assert new_command == 'apt list --upgradable'