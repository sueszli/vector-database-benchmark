from io import BytesIO
import pytest
from thefuck.types import Command
from thefuck.rules.apt_invalid_operation import match, get_new_command, _get_operations
invalid_operation = 'E: Invalid operation {}'.format
apt_help = b'apt 1.0.10.2ubuntu1 for amd64 compiled on Oct  5 2015 15:55:05\nUsage: apt [options] command\n\nCLI for apt.\nBasic commands:\n list - list packages based on package names\n search - search in package descriptions\n show - show package details\n\n update - update list of available packages\n\n install - install packages\n remove  - remove packages\n\n upgrade - upgrade the system by installing/upgrading packages\n full-upgrade - upgrade the system by removing/installing/upgrading packages\n\n edit-sources - edit the source information file\n'
apt_operations = ['list', 'search', 'show', 'update', 'install', 'remove', 'upgrade', 'full-upgrade', 'edit-sources']
apt_get_help = b'apt 1.0.10.2ubuntu1 for amd64 compiled on Oct  5 2015 15:55:05\nUsage: apt-get [options] command\n       apt-get [options] install|remove pkg1 [pkg2 ...]\n       apt-get [options] source pkg1 [pkg2 ...]\n\napt-get is a simple command line interface for downloading and\ninstalling packages. The most frequently used commands are update\nand install.\n\nCommands:\n   update - Retrieve new lists of packages\n   upgrade - Perform an upgrade\n   install - Install new packages (pkg is libc6 not libc6.deb)\n   remove - Remove packages\n   autoremove - Remove automatically all unused packages\n   purge - Remove packages and config files\n   source - Download source archives\n   build-dep - Configure build-dependencies for source packages\n   dist-upgrade - Distribution upgrade, see apt-get(8)\n   dselect-upgrade - Follow dselect selections\n   clean - Erase downloaded archive files\n   autoclean - Erase old downloaded archive files\n   check - Verify that there are no broken dependencies\n   changelog - Download and display the changelog for the given package\n   download - Download the binary package into the current directory\n\nOptions:\n  -h  This help text.\n  -q  Loggable output - no progress indicator\n  -qq No output except for errors\n  -d  Download only - do NOT install or unpack archives\n  -s  No-act. Perform ordering simulation\n  -y  Assume Yes to all queries and do not prompt\n  -f  Attempt to correct a system with broken dependencies in place\n  -m  Attempt to continue if archives are unlocatable\n  -u  Show a list of upgraded packages as well\n  -b  Build the source package after fetching it\n  -V  Show verbose version numbers\n  -c=? Read this configuration file\n  -o=? Set an arbitrary configuration option, eg -o dir::cache=/tmp\nSee the apt-get(8), sources.list(5) and apt.conf(5) manual\npages for more information and options.\n                       This APT has Super Cow Powers.\n'
apt_get_operations = ['update', 'upgrade', 'install', 'remove', 'autoremove', 'purge', 'source', 'build-dep', 'dist-upgrade', 'dselect-upgrade', 'clean', 'autoclean', 'check', 'changelog', 'download']
new_apt_get_help = b'apt 1.6.12 (amd64)\nUsage: apt-get [options] command\n       apt-get [options] install|remove pkg1 [pkg2 ...]\n       apt-get [options] source pkg1 [pkg2 ...]\n\napt-get is a command line interface for retrieval of packages\nand information about them from authenticated sources and\nfor installation, upgrade and removal of packages together\nwith their dependencies.\n\nMost used commands:\n  update - Retrieve new lists of packages\n  upgrade - Perform an upgrade\n  install - Install new packages (pkg is libc6 not libc6.deb)\n  remove - Remove packages\n  purge - Remove packages and config files\n  autoremove - Remove automatically all unused packages\n  dist-upgrade - Distribution upgrade, see apt-get(8)\n  dselect-upgrade - Follow dselect selections\n  build-dep - Configure build-dependencies for source packages\n  clean - Erase downloaded archive files\n  autoclean - Erase old downloaded archive files\n  check - Verify that there are no broken dependencies\n  source - Download source archives\n  download - Download the binary package into the current directory\n  changelog - Download and display the changelog for the given package\n\nSee apt-get(8) for more information about the available commands.\nConfiguration options and syntax is detailed in apt.conf(5).\nInformation about how to configure sources can be found in sources.list(5).\nPackage and version choices can be expressed via apt_preferences(5).\nSecurity details are available in apt-secure(8).\n                                        This APT has Super Cow Powers.\n'
new_apt_get_operations = ['update', 'upgrade', 'install', 'remove', 'purge', 'autoremove', 'dist-upgrade', 'dselect-upgrade', 'build-dep', 'clean', 'autoclean', 'check', 'source', 'download', 'changelog']

@pytest.mark.parametrize('script, output', [('apt', invalid_operation('saerch')), ('apt-get', invalid_operation('isntall')), ('apt-cache', invalid_operation('rumove'))])
def test_match(script, output):
    if False:
        for i in range(10):
            print('nop')
    assert match(Command(script, output))

@pytest.mark.parametrize('script, output', [('vim', invalid_operation('vim')), ('apt-get', '')])
def test_not_match(script, output):
    if False:
        print('Hello World!')
    assert not match(Command(script, output))

@pytest.fixture
def set_help(mocker):
    if False:
        while True:
            i = 10
    mock = mocker.patch('subprocess.Popen')

    def _set_text(text):
        if False:
            print('Hello World!')
        mock.return_value.stdout = BytesIO(text)
    return _set_text

@pytest.mark.parametrize('app, help_text, operations', [('apt', apt_help, apt_operations), ('apt-get', apt_get_help, apt_get_operations), ('apt-get', new_apt_get_help, new_apt_get_operations)])
def test_get_operations(set_help, app, help_text, operations):
    if False:
        i = 10
        return i + 15
    set_help(help_text)
    assert _get_operations(app) == operations

@pytest.mark.parametrize('script, output, help_text, result', [('apt-get isntall vim', invalid_operation('isntall'), apt_get_help, 'apt-get install vim'), ('apt saerch vim', invalid_operation('saerch'), apt_help, 'apt search vim'), ('apt uninstall vim', invalid_operation('uninstall'), apt_help, 'apt remove vim')])
def test_get_new_command(set_help, output, script, help_text, result):
    if False:
        print('Hello World!')
    set_help(help_text)
    assert get_new_command(Command(script, output))[0] == result