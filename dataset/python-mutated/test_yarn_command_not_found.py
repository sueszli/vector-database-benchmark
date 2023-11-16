from io import BytesIO
import pytest
from thefuck.types import Command
from thefuck.rules.yarn_command_not_found import match, get_new_command
output = '\nerror Command "{}" not found.\n'.format
yarn_help_stdout = b"\n\n  Usage: yarn [command] [flags]\n\n  Options:\n\n    -h, --help                      output usage information\n    -V, --version                   output the version number\n    --verbose                       output verbose messages on internal operations\n    --offline                       trigger an error if any required dependencies are not available in local cache\n    --prefer-offline                use network only if dependencies are not available in local cache\n    --strict-semver                 \n    --json                          \n    --ignore-scripts                don't run lifecycle scripts\n    --har                           save HAR output of network traffic\n    --ignore-platform               ignore platform checks\n    --ignore-engines                ignore engines check\n    --ignore-optional               ignore optional dependencies\n    --force                         ignore all caches\n    --no-bin-links                  don't generate bin links when setting up packages\n    --flat                          only allow one version of a package\n    --prod, --production [prod]     \n    --no-lockfile                   don't read or generate a lockfile\n    --pure-lockfile                 don't generate a lockfile\n    --frozen-lockfile               don't generate a lockfile and fail if an update is needed\n    --link-duplicates               create hardlinks to the repeated modules in node_modules\n    --global-folder <path>          \n    --modules-folder <path>         rather than installing modules into the node_modules folder relative to the cwd, output them here\n    --cache-folder <path>           specify a custom folder to store the yarn cache\n    --mutex <type>[:specifier]      use a mutex to ensure only one yarn instance is executing\n    --no-emoji                      disable emoji in output\n    --proxy <host>                  \n    --https-proxy <host>            \n    --no-progress                   disable progress bar\n    --network-concurrency <number>  maximum number of concurrent network requests\n\n  Commands:\n\n    - access\n    - add\n    - bin\n    - cache\n    - check\n    - clean\n    - config\n    - generate-lock-entry\n    - global\n    - import\n    - info\n    - init\n    - install\n    - licenses\n    - link\n    - list\n    - login\n    - logout\n    - outdated\n    - owner\n    - pack\n    - publish\n    - remove\n    - run\n    - tag\n    - team\n    - unlink\n    - upgrade\n    - upgrade-interactive\n    - version\n    - versions\n    - why\n\n  Run `yarn help COMMAND` for more information on specific commands.\n  Visit https://yarnpkg.com/en/docs/cli/ to learn more about Yarn.\n"

@pytest.fixture(autouse=True)
def yarn_help(mocker):
    if False:
        while True:
            i = 10
    patch = mocker.patch('thefuck.rules.yarn_command_not_found.Popen')
    patch.return_value.stdout = BytesIO(yarn_help_stdout)
    return patch

@pytest.mark.parametrize('command', [Command('yarn whyy webpack', output('whyy'))])
def test_match(command):
    if False:
        for i in range(10):
            print('nop')
    assert match(command)

@pytest.mark.parametrize('command', [Command('npm nuild', output('nuild')), Command('yarn install', '')])
def test_not_match(command):
    if False:
        i = 10
        return i + 15
    assert not match(command)

@pytest.mark.parametrize('command, result', [(Command('yarn whyy webpack', output('whyy')), 'yarn why webpack'), (Command('yarn require lodash', output('require')), 'yarn add lodash')])
def test_get_new_command(command, result):
    if False:
        print('Hello World!')
    fixed_command = get_new_command(command)
    if isinstance(fixed_command, list):
        fixed_command = fixed_command[0]
    assert fixed_command == result