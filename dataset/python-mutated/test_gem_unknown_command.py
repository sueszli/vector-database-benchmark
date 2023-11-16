import pytest
from six import BytesIO
from thefuck.rules.gem_unknown_command import match, get_new_command
from thefuck.types import Command
output = '\nERROR:  While executing gem ... (Gem::CommandLineError)\n    Unknown command {}\n'
gem_help_commands_stdout = b"\nGEM commands are:\n\n    build             Build a gem from a gemspec\n    cert              Manage RubyGems certificates and signing settings\n    check             Check a gem repository for added or missing files\n    cleanup           Clean up old versions of installed gems\n    contents          Display the contents of the installed gems\n    dependency        Show the dependencies of an installed gem\n    environment       Display information about the RubyGems environment\n    fetch             Download a gem and place it in the current directory\n    generate_index    Generates the index files for a gem server directory\n    help              Provide help on the 'gem' command\n    install           Install a gem into the local repository\n    list              Display local gems whose name matches REGEXP\n    lock              Generate a lockdown list of gems\n    mirror            Mirror all gem files (requires rubygems-mirror)\n    open              Open gem sources in editor\n    outdated          Display all gems that need updates\n    owner             Manage gem owners of a gem on the push server\n    pristine          Restores installed gems to pristine condition from files\n                      located in the gem cache\n    push              Push a gem up to the gem server\n    query             Query gem information in local or remote repositories\n    rdoc              Generates RDoc for pre-installed gems\n    search            Display remote gems whose name matches REGEXP\n    server            Documentation and gem repository HTTP server\n    sources           Manage the sources and cache file RubyGems uses to search\n                      for gems\n    specification     Display gem specification (in yaml)\n    stale             List gems along with access times\n    uninstall         Uninstall gems from the local repository\n    unpack            Unpack an installed gem to the current directory\n    update            Update installed gems to the latest version\n    which             Find the location of a library file you can require\n    yank              Remove a pushed gem from the index\n\nFor help on a particular command, use 'gem help COMMAND'.\n\nCommands may be abbreviated, so long as they are unambiguous.\ne.g. 'gem i rake' is short for 'gem install rake'.\n\n"

@pytest.fixture(autouse=True)
def gem_help_commands(mocker):
    if False:
        for i in range(10):
            print('nop')
    patch = mocker.patch('subprocess.Popen')
    patch.return_value.stdout = BytesIO(gem_help_commands_stdout)
    return patch

@pytest.mark.parametrize('script, command', [('gem isntall jekyll', 'isntall'), ('gem last --local', 'last')])
def test_match(script, command):
    if False:
        i = 10
        return i + 15
    assert match(Command(script, output.format(command)))

@pytest.mark.parametrize('script, output', [('gem install jekyll', ''), ('git log', output.format('log'))])
def test_not_match(script, output):
    if False:
        while True:
            i = 10
    assert not match(Command(script, output))

@pytest.mark.parametrize('script, output, result', [('gem isntall jekyll', output.format('isntall'), 'gem install jekyll'), ('gem last --local', output.format('last'), 'gem list --local')])
def test_get_new_command(script, output, result):
    if False:
        for i in range(10):
            print('nop')
    new_command = get_new_command(Command(script, output))
    assert new_command[0] == result