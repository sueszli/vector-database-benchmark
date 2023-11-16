import pytest
from thefuck.rules.npm_wrong_command import match, get_new_command
from thefuck.types import Command
output = '\nUsage: npm <command>\n\nwhere <command> is one of:\n    access, add-user, adduser, apihelp, author, bin, bugs, c,\n    cache, completion, config, ddp, dedupe, deprecate, dist-tag,\n    dist-tags, docs, edit, explore, faq, find, find-dupes, get,\n    help, help-search, home, i, info, init, install, issues, la,\n    link, list, ll, ln, login, logout, ls, outdated, owner,\n    pack, ping, prefix, prune, publish, r, rb, rebuild, remove,\n    repo, restart, rm, root, run-script, s, se, search, set,\n    show, shrinkwrap, star, stars, start, stop, t, tag, team,\n    test, tst, un, uninstall, unlink, unpublish, unstar, up,\n    update, upgrade, v, verison, version, view, whoami\n\nnpm <cmd> -h     quick help on <cmd>\nnpm -l           display full usage info\nnpm faq          commonly asked questions\nnpm help <term>  search for help on <term>\nnpm help npm     involved overview\n\nSpecify configs in the ini-formatted file:\n    /home/nvbn/.npmrc\nor on the command line via: npm <command> --key value\nConfig info can be viewed via: npm help config\n\nnpm@2.14.7 /opt/node/lib/node_modules/npm\n'

@pytest.mark.parametrize('script', ['npm urgrdae', 'npm urgrade -g', 'npm -f urgrade -g', 'npm urg'])
def test_match(script):
    if False:
        return 10
    assert match(Command(script, output))

@pytest.mark.parametrize('script, output', [('npm urgrade', ''), ('npm', output), ('test urgrade', output), ('npm -e', output)])
def test_not_match(script, output):
    if False:
        print('Hello World!')
    assert not match(Command(script, output))

@pytest.mark.parametrize('script, result', [('npm urgrade', 'npm upgrade'), ('npm -g isntall gulp', 'npm -g install gulp'), ('npm isntall -g gulp', 'npm install -g gulp')])
def test_get_new_command(script, result):
    if False:
        print('Hello World!')
    assert get_new_command(Command(script, output)) == result