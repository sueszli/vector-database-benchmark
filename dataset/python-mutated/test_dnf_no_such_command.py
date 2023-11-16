from io import BytesIO
import pytest
from thefuck.types import Command
from thefuck.rules.dnf_no_such_command import match, get_new_command, _get_operations
help_text = b"usage: dnf [options] COMMAND\n\nList of Main Commands:\n\nautoremove                remove all unneeded packages that were originally installed as dependencies\ncheck                     check for problems in the packagedb\ncheck-update              check for available package upgrades\nclean                     remove cached data\ndeplist                   List package's dependencies and what packages provide them\ndistro-sync               synchronize installed packages to the latest available versions\ndowngrade                 Downgrade a package\ngroup                     display, or use, the groups information\nhelp                      display a helpful usage message\nhistory                   display, or use, the transaction history\ninfo                      display details about a package or group of packages\ninstall                   install a package or packages on your system\nlist                      list a package or groups of packages\nmakecache                 generate the metadata cache\nmark                      mark or unmark installed packages as installed by user.\nprovides                  find what package provides the given value\nreinstall                 reinstall a package\nremove                    remove a package or packages from your system\nrepolist                  display the configured software repositories\nrepoquery                 search for packages matching keyword\nrepository-packages       run commands on top of all packages in given repository\nsearch                    search package details for the given string\nshell                     run an interactive DNF shell\nswap                      run an interactive dnf mod for remove and install one spec\nupdateinfo                display advisories about packages\nupgrade                   upgrade a package or packages on your system\nupgrade-minimal           upgrade, but only 'newest' package match which fixes a problem that affects your system\n\nList of Plugin Commands:\n\nbuilddep                  Install build dependencies for package or spec file\nconfig-manager            manage dnf configuration options and repositories\ncopr                      Interact with Copr repositories.\ndebug-dump                dump information about installed rpm packages to file\ndebug-restore             restore packages recorded in debug-dump file\ndebuginfo-install         install debuginfo packages\ndownload                  Download package to current directory\nneeds-restarting          determine updated binaries that need restarting\nplayground                Interact with Playground repository.\nrepoclosure               Display a list of unresolved dependencies for repositories\nrepograph                 Output a full package dependency graph in dot format\nrepomanage                Manage a directory of rpm packages\nreposync                  download all packages from remote repo\n\nOptional arguments:\n  -c [config file], --config [config file]\n                        config file location\n  -q, --quiet           quiet operation\n  -v, --verbose         verbose operation\n  --version             show DNF version and exit\n  --installroot [path]  set install root\n  --nodocs              do not install documentations\n  --noplugins           disable all plugins\n  --enableplugin [plugin]\n                        enable plugins by name\n  --disableplugin [plugin]\n                        disable plugins by name\n  --releasever RELEASEVER\n                        override the value of $releasever in config and repo\n                        files\n  --setopt SETOPTS      set arbitrary config and repo options\n  --skip-broken         resolve depsolve problems by skipping packages\n  -h, --help, --help-cmd\n                        show command help\n  --allowerasing        allow erasing of installed packages to resolve\n                        dependencies\n  -b, --best            try the best available package versions in\n                        transactions.\n  -C, --cacheonly       run entirely from system cache, don't update cache\n  -R [minutes], --randomwait [minutes]\n                        maximum command wait time\n  -d [debug level], --debuglevel [debug level]\n                        debugging output level\n  --debugsolver         dumps detailed solving results into files\n  --showduplicates      show duplicates, in repos, in list/search commands\n  -e ERRORLEVEL, --errorlevel ERRORLEVEL\n                        error output level\n  --obsoletes           enables dnf's obsoletes processing logic for upgrade\n                        or display capabilities that the package obsoletes for\n                        info, list and repoquery\n  --rpmverbosity [debug level name]\n                        debugging output level for rpm\n  -y, --assumeyes       automatically answer yes for all questions\n  --assumeno            automatically answer no for all questions\n  --enablerepo [repo]\n  --disablerepo [repo]\n  --repo [repo], --repoid [repo]\n                        enable just specific repositories by an id or a glob,\n                        can be specified multiple times\n  -x [package], --exclude [package], --excludepkgs [package]\n                        exclude packages by name or glob\n  --disableexcludes [repo], --disableexcludepkgs [repo]\n                        disable excludepkgs\n  --repofrompath [repo,path]\n                        label and path to additional repository, can be\n                        specified multiple times.\n  --noautoremove        disable removal of dependencies that are no longer\n                        used\n  --nogpgcheck          disable gpg signature checking\n  --color COLOR         control whether colour is used\n  --refresh             set metadata as expired before running the command\n  -4                    resolve to IPv4 addresses only\n  -6                    resolve to IPv6 addresses only\n  --destdir DESTDIR, --downloaddir DESTDIR\n                        set directory to copy packages to\n  --downloadonly        only download packages\n  --bugfix              Include bugfix relevant packages, in updates\n  --enhancement         Include enhancement relevant packages, in updates\n  --newpackage          Include newpackage relevant packages, in updates\n  --security            Include security relevant packages, in updates\n  --advisory ADVISORY, --advisories ADVISORY\n                        Include packages needed to fix the given advisory, in\n                        updates\n  --bzs BUGZILLA        Include packages needed to fix the given BZ, in\n                        updates\n  --cves CVES           Include packages needed to fix the given CVE, in\n                        updates\n  --sec-severity {Critical,Important,Moderate,Low}, --secseverity {Critical,Important,Moderate,Low}\n                        Include security relevant packages matching the\n                        severity, in updates\n  --forcearch ARCH      Force the use of an architecture\n"
dnf_operations = ['autoremove', 'check', 'check-update', 'clean', 'deplist', 'distro-sync', 'downgrade', 'group', 'help', 'history', 'info', 'install', 'list', 'makecache', 'mark', 'provides', 'reinstall', 'remove', 'repolist', 'repoquery', 'repository-packages', 'search', 'shell', 'swap', 'updateinfo', 'upgrade', 'upgrade-minimal', 'builddep', 'config-manager', 'copr', 'debug-dump', 'debug-restore', 'debuginfo-install', 'download', 'needs-restarting', 'playground', 'repoclosure', 'repograph', 'repomanage', 'reposync']

def invalid_command(command):
    if False:
        i = 10
        return i + 15
    return 'No such command: %s. Please use /usr/bin/dnf --help\nIt could be a DNF plugin command, try: "dnf install \'dnf-command(%s)\'"\n' % (command, command)

@pytest.mark.parametrize('output', [invalid_command('saerch'), invalid_command('isntall')])
def test_match(output):
    if False:
        for i in range(10):
            print('nop')
    assert match(Command('dnf', output))

@pytest.mark.parametrize('script, output', [('pip', invalid_command('isntall')), ('vim', '')])
def test_not_match(script, output):
    if False:
        return 10
    assert not match(Command(script, output))

@pytest.fixture
def set_help(mocker):
    if False:
        i = 10
        return i + 15
    mock = mocker.patch('subprocess.Popen')

    def _set_text(text):
        if False:
            i = 10
            return i + 15
        mock.return_value.stdout = BytesIO(text)
    return _set_text

def test_get_operations(set_help):
    if False:
        while True:
            i = 10
    set_help(help_text)
    assert _get_operations() == dnf_operations

@pytest.mark.parametrize('script, output, result', [('dnf isntall vim', invalid_command('isntall'), 'dnf install vim'), ('dnf saerch vim', invalid_command('saerch'), 'dnf search vim')])
def test_get_new_command(set_help, output, script, result):
    if False:
        while True:
            i = 10
    set_help(help_text)
    assert result in get_new_command(Command(script, output))