import pytest
from io import BytesIO
from thefuck.rules.npm_run_script import match, get_new_command
from thefuck.types import Command
output = '\nUsage: npm <command>\n\nwhere <command> is one of:\n    access, add-user, adduser, apihelp, author, bin, bugs, c,\n    cache, completion, config, ddp, dedupe, deprecate, dist-tag,\n    dist-tags, docs, edit, explore, faq, find, find-dupes, get,\n    help, help-search, home, i, info, init, install, issues, la,\n    link, list, ll, ln, login, logout, ls, outdated, owner,\n    pack, ping, prefix, prune, publish, r, rb, rebuild, remove,\n    repo, restart, rm, root, run-script, s, se, search, set,\n    show, shrinkwrap, star, stars, start, stop, t, tag, team,\n    test, tst, un, uninstall, unlink, unpublish, unstar, up,\n    update, upgrade, v, version, view, whoami\n\nnpm <cmd> -h     quick help on <cmd>\nnpm -l           display full usage info\nnpm faq          commonly asked questions\nnpm help <term>  search for help on <term>\nnpm help npm     involved overview\n\nSpecify configs in the ini-formatted file:\n    /home/nvbn/.npmrc\nor on the command line via: npm <command> --key value\nConfig info can be viewed via: npm help config\n\n'
run_script_stdout = b'\nLifecycle scripts included in code-view-web:\n  test\n    jest\n\navailable via `npm run-script`:\n  build\n    cp node_modules/ace-builds/src-min/ -a resources/ace/ && webpack --progress --colors -p --config ./webpack.production.config.js\n  develop\n    cp node_modules/ace-builds/src/ -a resources/ace/ && webpack-dev-server --progress --colors\n  watch-test\n    jest --verbose --watch\n\n'

@pytest.fixture(autouse=True)
def run_script(mocker):
    if False:
        return 10
    patch = mocker.patch('thefuck.specific.npm.Popen')
    patch.return_value.stdout = BytesIO(run_script_stdout)
    return patch.return_value

@pytest.mark.usefixtures('no_memoize')
@pytest.mark.parametrize('script', ['npm watch-test', 'npm develop'])
def test_match(script):
    if False:
        print('Hello World!')
    command = Command(script, output)
    assert match(command)

@pytest.mark.usefixtures('no_memoize')
@pytest.mark.parametrize('command, run_script_out', [(Command('npm test', 'TEST FAIL'), run_script_stdout), (Command('npm watch-test', 'TEST FAIL'), run_script_stdout), (Command('npm test', output), run_script_stdout), (Command('vim watch-test', output), run_script_stdout)])
def test_not_match(run_script, command, run_script_out):
    if False:
        while True:
            i = 10
    run_script.stdout = BytesIO(run_script_out)
    assert not match(command)

@pytest.mark.usefixtures('no_memoize')
@pytest.mark.parametrize('script, result', [('npm watch-test', 'npm run-script watch-test'), ('npm -i develop', 'npm run-script -i develop'), ('npm -i watch-script --path ..', 'npm run-script -i watch-script --path ..')])
def test_get_new_command(script, result):
    if False:
        while True:
            i = 10
    command = Command(script, output)
    assert get_new_command(command) == result