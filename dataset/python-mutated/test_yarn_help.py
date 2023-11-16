import pytest
from thefuck.rules.yarn_help import match, get_new_command
from thefuck.types import Command
from thefuck.system import open_command
output_clean = "\n\n  Usage: yarn [command] [flags]\n\n  Options:\n\n    -h, --help                      output usage information\n    -V, --version                   output the version number\n    --verbose                       output verbose messages on internal operations\n    --offline                       trigger an error if any required dependencies are not available in local cache\n    --prefer-offline                use network only if dependencies are not available in local cache\n    --strict-semver                 \n    --json                          \n    --ignore-scripts                don't run lifecycle scripts\n    --har                           save HAR output of network traffic\n    --ignore-platform               ignore platform checks\n    --ignore-engines                ignore engines check\n    --ignore-optional               ignore optional dependencies\n    --force                         ignore all caches\n    --no-bin-links                  don't generate bin links when setting up packages\n    --flat                          only allow one version of a package\n    --prod, --production [prod]     \n    --no-lockfile                   don't read or generate a lockfile\n    --pure-lockfile                 don't generate a lockfile\n    --frozen-lockfile               don't generate a lockfile and fail if an update is needed\n    --link-duplicates               create hardlinks to the repeated modules in node_modules\n    --global-folder <path>          \n    --modules-folder <path>         rather than installing modules into the node_modules folder relative to the cwd, output them here\n    --cache-folder <path>           specify a custom folder to store the yarn cache\n    --mutex <type>[:specifier]      use a mutex to ensure only one yarn instance is executing\n    --no-emoji                      disable emoji in output\n    --proxy <host>                  \n    --https-proxy <host>            \n    --no-progress                   disable progress bar\n    --network-concurrency <number>  maximum number of concurrent network requests\n\n  Visit https://yarnpkg.com/en/docs/cli/clean for documentation about this command.\n"

@pytest.mark.parametrize('command', [Command('yarn help clean', output_clean)])
def test_match(command):
    if False:
        return 10
    assert match(command)

@pytest.mark.parametrize('command, url', [(Command('yarn help clean', output_clean), 'https://yarnpkg.com/en/docs/cli/clean')])
def test_get_new_command(command, url):
    if False:
        return 10
    assert get_new_command(command) == open_command(url)