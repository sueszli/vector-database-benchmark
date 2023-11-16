import pytest
from io import BytesIO
from thefuck.types import Command
from thefuck.rules.npm_missing_script import match, get_new_command
output = '\nnpm ERR! Linux 4.4.0-31-generic\nnpm ERR! argv "/opt/node/bin/node" "/opt/node/bin/npm" "run" "dvelop"\nnpm ERR! node v4.4.7\nnpm ERR! npm  v2.15.8\n\nnpm ERR! missing script: {}\nnpm ERR!\nnpm ERR! If you need help, you may report this error at:\nnpm ERR!     <https://github.com/npm/npm/issues>\n\nnpm ERR! Please include the following file with any support request:\nnpm ERR!     /home/nvbn/exp/code_view/client_web/npm-debug.log\n'.format
run_script_stdout = b'\nLifecycle scripts included in code-view-web:\n  test\n    jest\n\navailable via `npm run-script`:\n  build\n    cp node_modules/ace-builds/src-min/ -a resources/ace/ && webpack --progress --colors -p --config ./webpack.production.config.js\n  develop\n    cp node_modules/ace-builds/src/ -a resources/ace/ && webpack-dev-server --progress --colors\n  watch-test\n    jest --verbose --watch\n\n'

@pytest.fixture(autouse=True)
def run_script(mocker):
    if False:
        while True:
            i = 10
    patch = mocker.patch('thefuck.specific.npm.Popen')
    patch.return_value.stdout = BytesIO(run_script_stdout)
    return patch.return_value

@pytest.mark.parametrize('command', [Command('npm ru wach', output('wach')), Command('npm run live-tes', output('live-tes')), Command('npm run-script sahare', output('sahare'))])
def test_match(command):
    if False:
        while True:
            i = 10
    assert match(command)

@pytest.mark.parametrize('command', [Command('npm wach', output('wach')), Command('vim live-tes', output('live-tes')), Command('npm run-script sahare', '')])
def test_not_match(command):
    if False:
        return 10
    assert not match(command)

@pytest.mark.parametrize('script, output, result', [('npm ru wach-tests', output('wach-tests'), 'npm ru watch-test'), ('npm -i run-script dvelop', output('dvelop'), 'npm -i run-script develop'), ('npm -i run-script buld -X POST', output('buld'), 'npm -i run-script build -X POST')])
def test_get_new_command(script, output, result):
    if False:
        i = 10
        return i + 15
    command = Command(script, output)
    assert get_new_command(command)[0] == result