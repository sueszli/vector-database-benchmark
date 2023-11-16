from io import BytesIO
import pytest
from thefuck.rules.yum_invalid_operation import match, get_new_command, _get_operations
from thefuck.types import Command
yum_help_text = "Loaded plugins: extras_suggestions, langpacks, priorities, update-motd\nUsage: yum [options] COMMAND\n\nList of Commands:\n\ncheck          Check for problems in the rpmdb\ncheck-update   Check for available package updates\nclean          Remove cached data\ndeplist        List a package's dependencies\ndistribution-synchronization Synchronize installed packages to the latest available versions\ndowngrade      downgrade a package\nerase          Remove a package or packages from your system\nfs             Acts on the filesystem data of the host, mainly for removing docs/lanuages for minimal hosts.\nfssnapshot     Creates filesystem snapshots, or lists/deletes current snapshots.\ngroups         Display, or use, the groups information\nhelp           Display a helpful usage message\nhistory        Display, or use, the transaction history\ninfo           Display details about a package or group of packages\ninstall        Install a package or packages on your system\nlangavailable  Check available languages\nlanginfo       List languages information\nlanginstall    Install appropriate language packs for a language\nlanglist       List installed languages\nlangremove     Remove installed language packs for a language\nlist           List a package or groups of packages\nload-transaction load a saved transaction from filename\nmakecache      Generate the metadata cache\nprovides       Find what package provides the given value\nreinstall      reinstall a package\nrepo-pkgs      Treat a repo. as a group of packages, so we can install/remove all of them\nrepolist       Display the configured software repositories\nsearch         Search package details for the given string\nshell          Run an interactive yum shell\nswap           Simple way to swap packages, instead of using shell\nupdate         Update a package or packages on your system\nupdate-minimal Works like upgrade, but goes to the 'newest' package match which fixes a problem that affects your system\nupdateinfo     Acts on repository update information\nupgrade        Update packages taking obsoletes into account\nversion        Display a version for the machine and/or available repos.\n\n\nOptions:\n  -h, --help            show this help message and exit\n  -t, --tolerant        be tolerant of errors\n  -C, --cacheonly       run entirely from system cache, don't update cache\n  -c [config file], --config=[config file]\n                        config file location\n  -R [minutes], --randomwait=[minutes]\n                        maximum command wait time\n  -d [debug level], --debuglevel=[debug level]\n                        debugging output level\n  --showduplicates      show duplicates, in repos, in list/search commands\n  -e [error level], --errorlevel=[error level]\n                        error output level\n  --rpmverbosity=[debug level name]\n                        debugging output level for rpm\n  -q, --quiet           quiet operation\n  -v, --verbose         verbose operation\n  -y, --assumeyes       answer yes for all questions\n  --assumeno            answer no for all questions\n  --version             show Yum version and exit\n  --installroot=[path]  set install root\n  --enablerepo=[repo]   enable one or more repositories (wildcards allowed)\n  --disablerepo=[repo]  disable one or more repositories (wildcards allowed)\n  -x [package], --exclude=[package]\n                        exclude package(s) by name or glob\n  --disableexcludes=[repo]\n                        disable exclude from main, for a repo or for\n                        everything\n  --disableincludes=[repo]\n                        disable includepkgs for a repo or for everything\n  --obsoletes           enable obsoletes processing during updates\n  --noplugins           disable Yum plugins\n  --nogpgcheck          disable gpg signature checking\n  --disableplugin=[plugin]\n                        disable plugins by name\n  --enableplugin=[plugin]\n                        enable plugins by name\n  --skip-broken         skip packages with depsolving problems\n  --color=COLOR         control whether color is used\n  --releasever=RELEASEVER\n                        set value of $releasever in yum config and repo files\n  --downloadonly        don't update, just download\n  --downloaddir=DLDIR   specifies an alternate directory to store packages\n  --setopt=SETOPTS      set arbitrary config and repo options\n  --bugfix              Include bugfix relevant packages, in updates\n  --security            Include security relevant packages, in updates\n  --advisory=ADVS, --advisories=ADVS\n                        Include packages needed to fix the given advisory, in\n                        updates\n  --bzs=BZS             Include packages needed to fix the given BZ, in\n                        updates\n  --cves=CVES           Include packages needed to fix the given CVE, in\n                        updates\n  --sec-severity=SEVS, --secseverity=SEVS\n                        Include security relevant packages matching the\n                        severity, in updates\n\n  Plugin Options:\n    --samearch-priorities\n                        Priority-exclude packages based on name + arch\n"
yum_unsuccessful_search_text = 'Warning: No matches found for: {}\nNo matches found\n'
yum_successful_vim_search_text = '================================================== N/S matched: vim ===================================================\nprotobuf-vim.x86_64 : Vim syntax highlighting for Google Protocol Buffers descriptions\nvim-X11.x86_64 : The VIM version of the vi editor for the X Window System - GVim\nvim-common.x86_64 : The common files needed by any version of the VIM editor\nvim-enhanced.x86_64 : A version of the VIM editor which includes recent enhancements\nvim-filesystem.x86_64 : VIM filesystem layout\nvim-filesystem.noarch : VIM filesystem layout\nvim-minimal.x86_64 : A minimal version of the VIM editor\n\n  Name and summary matches only, use "search all" for everything.\n'
yum_invalid_op_text = 'Loaded plugins: extras_suggestions, langpacks, priorities, update-motd\nNo such command: {}. Please use /usr/bin/yum --help\n'
yum_operations = ['check', 'check-update', 'clean', 'deplist', 'distribution-synchronization', 'downgrade', 'erase', 'fs', 'fssnapshot', 'groups', 'help', 'history', 'info', 'install', 'langavailable', 'langinfo', 'langinstall', 'langlist', 'langremove', 'list', 'load-transaction', 'makecache', 'provides', 'reinstall', 'repo-pkgs', 'repolist', 'search', 'shell', 'swap', 'update', 'update-minimal', 'updateinfo', 'upgrade', 'version']

@pytest.mark.parametrize('command', ['saerch', 'uninstall'])
def test_match(command):
    if False:
        i = 10
        return i + 15
    assert match(Command('yum {}'.format(command), yum_invalid_op_text.format(command)))

@pytest.mark.parametrize('command, output', [('vim', ''), ('yum', yum_help_text), ('yum help', yum_help_text), ('yum search asdf', yum_unsuccessful_search_text.format('asdf')), ('yum search vim', yum_successful_vim_search_text)])
def test_not_match(command, output):
    if False:
        i = 10
        return i + 15
    assert not match(Command(command, output))

@pytest.fixture
def yum_help(mocker):
    if False:
        return 10
    mock = mocker.patch('subprocess.Popen')
    mock.return_value.stdout = BytesIO(bytes(yum_help_text.encode('utf-8')))
    return mock

@pytest.mark.usefixtures('no_memoize', 'yum_help')
def test_get_operations():
    if False:
        return 10
    assert _get_operations() == yum_operations

@pytest.mark.usefixtures('no_memoize', 'yum_help')
@pytest.mark.parametrize('script, output, result', [('yum uninstall', yum_invalid_op_text.format('uninstall'), 'yum remove'), ('yum saerch asdf', yum_invalid_op_text.format('saerch'), 'yum search asdf'), ('yum hlep', yum_invalid_op_text.format('hlep'), 'yum help')])
def test_get_new_command(script, output, result):
    if False:
        for i in range(10):
            print('nop')
    assert get_new_command(Command(script, output))[0] == result