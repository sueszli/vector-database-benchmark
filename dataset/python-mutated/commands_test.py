import logging
import os
import re
import tarfile
from os import mkdir, path
from queue import Empty
from shutil import rmtree
from tempfile import mkdtemp
import pytest
from mock import MagicMock
extra_plugin_dir = path.join(path.dirname(path.realpath(__file__)), 'dummy_plugin')

def test_root_help(testbot):
    if False:
        while True:
            i = 10
    assert 'All commands' in testbot.exec_command('!help')

def test_help(testbot):
    if False:
        return 10
    assert '!about' in testbot.exec_command('!help Help')
    assert 'That command is not defined.' in testbot.exec_command('!help beurk')
    assert 'runs foo' in testbot.exec_command('!help foo')
    assert 'runs re_foo' in testbot.exec_command('!help re_foo')
    assert 'runs re_foo' in testbot.exec_command('!help re foo')

def test_about(testbot):
    if False:
        while True:
            i = 10
    assert 'Errbot version' in testbot.exec_command('!about')

def test_uptime(testbot):
    if False:
        i = 10
        return i + 15
    assert "I've been up for" in testbot.exec_command('!uptime')

def test_status(testbot):
    if False:
        for i in range(10):
            print('nop')
    assert 'Yes I am alive' in testbot.exec_command('!status')

def test_status_plugins(testbot):
    if False:
        print('Hello World!')
    assert 'A = Activated, D = Deactivated' in testbot.exec_command('!status plugins')

def test_status_load(testbot):
    if False:
        while True:
            i = 10
    assert 'Load ' in testbot.exec_command('!status load')

def test_whoami(testbot):
    if False:
        print('Hello World!')
    assert 'person' in testbot.exec_command('!whoami')
    assert 'gbin@localhost' in testbot.exec_command('!whoami')

def test_echo(testbot):
    if False:
        while True:
            i = 10
    assert 'foo' in testbot.exec_command('!echo foo')

def test_status_gc(testbot):
    if False:
        print('Hello World!')
    assert 'GC 0->' in testbot.exec_command('!status gc')

def test_config_cycle(testbot):
    if False:
        print('Hello World!')
    testbot.push_message('!plugin config Webserver')
    m = testbot.pop_message()
    assert 'Default configuration for this plugin (you can copy and paste this directly as a command)' in m
    assert 'Current configuration' not in m
    testbot.assertInCommand("!plugin config Webserver {'HOST': 'localhost', 'PORT': 3141, 'SSL':  None}", 'Plugin configuration done.')
    assert 'Current configuration' in testbot.exec_command('!plugin config Webserver')
    assert 'localhost' in testbot.exec_command('!plugin config Webserver')

def test_apropos(testbot):
    if False:
        for i in range(10):
            print('nop')
    assert '!about: Return information about' in testbot.exec_command('!apropos about')

def test_logtail(testbot):
    if False:
        return 10
    assert 'DEBUG' in testbot.exec_command('!log tail')

def test_history(testbot):
    if False:
        return 10
    assert 'up' in testbot.exec_command('!uptime')
    assert 'uptime' in testbot.exec_command('!history')
    orig_sender = testbot.bot.sender
    testbot.bot.sender = testbot.bot.build_identifier('non_default_person')
    testbot.push_message('!history')
    with pytest.raises(Empty):
        testbot.pop_message(timeout=1)
    assert 'should be a separate history' in testbot.exec_command('!echo should be a separate history')
    assert 'should be a separate history' in testbot.exec_command('!history')
    testbot.bot.sender = orig_sender
    assert 'uptime' in testbot.exec_command('!history')

def test_plugin_cycle(testbot):
    if False:
        for i in range(10):
            print('nop')
    plugins = ['errbotio/err-helloworld']
    for plugin in plugins:
        (testbot.assertInCommand(f'!repos install {plugin}', f'Installing {plugin}...'),)
        assert 'A new plugin repository has been installed correctly from errbotio/err-helloworld' in testbot.pop_message(timeout=60)
        assert 'Plugins reloaded' in testbot.pop_message()
        assert 'this command says hello' in testbot.exec_command('!help hello')
        assert 'Hello World !' in testbot.exec_command('!hello')
        testbot.push_message('!plugin reload HelloWorld')
        assert 'Plugin HelloWorld reloaded.' == testbot.pop_message()
        testbot.push_message('!hello')
        assert 'Hello World !' == testbot.pop_message()
        testbot.push_message('!plugin blacklist HelloWorld')
        assert 'Plugin HelloWorld is now blacklisted.' == testbot.pop_message()
        testbot.push_message('!plugin deactivate HelloWorld')
        assert 'HelloWorld is already deactivated.' == testbot.pop_message()
        testbot.push_message('!hello')
        assert 'Command "hello" not found' in testbot.pop_message()
        testbot.push_message('!plugin unblacklist HelloWorld')
        assert 'Plugin HelloWorld removed from blacklist.' == testbot.pop_message()
        testbot.push_message('!plugin activate HelloWorld')
        assert 'HelloWorld is already activated.' == testbot.pop_message()
        testbot.push_message('!hello')
        assert 'Hello World !' == testbot.pop_message()
        testbot.push_message('!repos uninstall errbotio/err-helloworld')
        assert 'Repo errbotio/err-helloworld removed.' == testbot.pop_message()
        testbot.push_message('!hello')
        assert 'Command "hello" not found' in testbot.pop_message()

def test_broken_plugin(testbot):
    if False:
        return 10
    borken_plugin_dir = path.join(path.dirname(path.realpath(__file__)), 'borken_plugin')
    try:
        tempd = mkdtemp()
        tgz = os.path.join(tempd, 'borken.tar.gz')
        with tarfile.open(tgz, 'w:gz') as tar:
            tar.add(borken_plugin_dir, arcname='borken')
        assert 'Installing' in testbot.exec_command('!repos install file://' + tgz, timeout=120)
        assert 'import borken  # fails' in testbot.pop_message()
        assert 'as it did not load correctly.' in testbot.pop_message()
        assert "Error: Broken failed to activate: 'NoneType' object has no attribute 'is_activated'" in testbot.pop_message()
        assert 'Plugins reloaded.' in testbot.pop_message()
    finally:
        rmtree(tempd)

def test_backup(testbot):
    if False:
        print('Hello World!')
    bot = testbot.bot
    bot.push_message('!repos install https://github.com/errbotio/err-helloworld.git')
    assert 'Installing' in testbot.pop_message()
    assert 'err-helloworld' in testbot.pop_message(timeout=60)
    assert 'reload' in testbot.pop_message()
    bot.push_message('!backup')
    msg = testbot.pop_message()
    assert 'has been written in' in msg
    filename = re.search('"(.*)"', msg).group(1)
    assert 'errbotio/err-helloworld' in open(filename).read()
    for p in testbot.bot.plugin_manager.get_all_active_plugins():
        p.close_storage()
    assert 'Plugin HelloWorld deactivated.' in testbot.exec_command('!plugin deactivate HelloWorld')
    plugins_dir = path.join(testbot.bot_config.BOT_DATA_DIR, 'plugins')
    bot.repo_manager['installed_repos'] = {}
    bot.plugin_manager['configs'] = {}
    rmtree(plugins_dir)
    mkdir(plugins_dir)
    from errbot.bootstrap import restore_bot_from_backup
    log = logging.getLogger(__name__)
    restore_bot_from_backup(filename, bot=bot, log=log)
    assert 'Plugin HelloWorld activated.' in testbot.exec_command('!plugin activate HelloWorld')
    assert 'Hello World !' in testbot.exec_command('!hello')
    testbot.push_message('!repos uninstall errbotio/err-helloworld')

def test_encoding_preservation(testbot):
    if False:
        while True:
            i = 10
    testbot.push_message('!echo へようこそ')
    assert 'へようこそ' == testbot.pop_message()

def test_webserver_webhook_test(testbot):
    if False:
        print('Hello World!')
    testbot.push_message("!plugin config Webserver {'HOST': 'localhost', 'PORT': 3141, 'SSL':  None}")
    assert 'Plugin configuration done.' in testbot.pop_message()
    testbot.assertInCommand('!webhook test /echo toto', 'Status code: 200')

def test_activate_reload_and_deactivate(testbot):
    if False:
        while True:
            i = 10
    for command in ('activate', 'reload', 'deactivate'):
        testbot.push_message(f'!plugin {command}')
        m = testbot.pop_message()
        assert 'Please tell me which of the following plugins to' in m
        assert 'ChatRoom' in m
        testbot.push_message(f'!plugin {command} nosuchplugin')
        m = testbot.pop_message()
        assert "nosuchplugin isn't a valid plugin name. The current plugins are" in m
        assert 'ChatRoom' in m
    testbot.push_message('!plugin reload ChatRoom')
    assert 'Plugin ChatRoom reloaded.' == testbot.pop_message()
    testbot.push_message('!status plugins')
    assert 'A      │ ChatRoom' in testbot.pop_message()
    testbot.push_message('!plugin deactivate ChatRoom')
    assert 'Plugin ChatRoom deactivated.' == testbot.pop_message()
    testbot.push_message('!status plugins')
    assert 'D      │ ChatRoom' in testbot.pop_message()
    testbot.push_message('!plugin deactivate ChatRoom')
    assert 'ChatRoom is already deactivated.' in testbot.pop_message()
    testbot.push_message('!plugin activate ChatRoom')
    assert 'Plugin ChatRoom activated.' in testbot.pop_message()
    testbot.push_message('!status plugins')
    assert 'A      │ ChatRoom' in testbot.pop_message()
    testbot.push_message('!plugin activate ChatRoom')
    assert 'ChatRoom is already activated.' == testbot.pop_message()
    testbot.push_message('!plugin deactivate ChatRoom')
    assert 'Plugin ChatRoom deactivated.' == testbot.pop_message()
    testbot.push_message('!plugin reload ChatRoom')
    assert 'Warning: plugin ChatRoom is currently not activated. Use !plugin activate ChatRoom to activate it.' == testbot.pop_message()
    assert 'Plugin ChatRoom reloaded.' == testbot.pop_message()
    testbot.push_message('!plugin blacklist ChatRoom')
    assert 'Plugin ChatRoom is now blacklisted.' == testbot.pop_message()
    testbot.push_message('!status plugins')
    assert 'B,D    │ ChatRoom' in testbot.pop_message()
    testbot.push_message('!plugin unblacklist ChatRoom')
    testbot.pop_message()

def test_unblacklist_and_blacklist(testbot):
    if False:
        for i in range(10):
            print('nop')
    testbot.push_message('!plugin unblacklist nosuchplugin')
    m = testbot.pop_message()
    assert "nosuchplugin isn't a valid plugin name. The current plugins are" in m
    assert 'ChatRoom' in m
    testbot.push_message('!plugin blacklist nosuchplugin')
    m = testbot.pop_message()
    assert "nosuchplugin isn't a valid plugin name. The current plugins are" in m
    assert 'ChatRoom' in m
    testbot.push_message('!plugin blacklist ChatRoom')
    assert 'Plugin ChatRoom is now blacklisted' in testbot.pop_message()
    testbot.push_message('!plugin blacklist ChatRoom')
    assert 'Plugin ChatRoom is already blacklisted.' == testbot.pop_message()
    testbot.push_message('!status plugins')
    assert 'B,D    │ ChatRoom' in testbot.pop_message()
    testbot.push_message('!plugin unblacklist ChatRoom')
    assert 'Plugin ChatRoom removed from blacklist.' == testbot.pop_message()
    testbot.push_message('!plugin unblacklist ChatRoom')
    assert 'Plugin ChatRoom is not blacklisted.' == testbot.pop_message()
    testbot.push_message('!status plugins')
    assert 'A      │ ChatRoom' in testbot.pop_message()

def test_optional_prefix(testbot):
    if False:
        for i in range(10):
            print('nop')
    testbot.bot_config.BOT_PREFIX_OPTIONAL_ON_CHAT = False
    assert 'Yes I am alive' in testbot.exec_command('!status')
    testbot.bot_config.BOT_PREFIX_OPTIONAL_ON_CHAT = True
    assert 'Yes I am alive' in testbot.exec_command('!status')
    assert 'Yes I am alive' in testbot.exec_command('status')

def test_optional_prefix_re_cmd(testbot):
    if False:
        while True:
            i = 10
    testbot.bot_config.BOT_PREFIX_OPTIONAL_ON_CHAT = False
    assert 'bar' in testbot.exec_command('!plz dont match this')
    testbot.bot_config.BOT_PREFIX_OPTIONAL_ON_CHAT = True
    assert 'bar' in testbot.exec_command('!plz dont match this')
    assert 'bar' in testbot.exec_command('plz dont match this')

def test_simple_match(testbot):
    if False:
        print('Hello World!')
    assert 'bar' in testbot.exec_command('match this')

def test_no_suggest_on_re_commands(testbot):
    if False:
        i = 10
        return i + 15
    testbot.push_message('!re_ba')
    assert '!re bar' not in testbot.pop_message()

def test_callback_no_command(testbot):
    if False:
        i = 10
        return i + 15
    extra_plugin_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'commandnotfound_plugin')
    cmd = '!this_is_not_a_real_command_at_all'
    expected_str = f'Command fell through: {cmd}'
    testbot.exec_command('!plugin deactivate CommandNotFoundFilter')
    testbot.bot.plugin_manager._extra_plugin_dir = extra_plugin_dir
    testbot.bot.plugin_manager.update_plugin_places([])
    testbot.exec_command('!plugin activate TestCommandNotFoundFilter')
    assert expected_str == testbot.exec_command(cmd)

def test_subcommands(testbot):
    if False:
        while True:
            i = 10
    cmd = '!run subcommands with these args'
    cmd_underscore = '!run_subcommands with these args'
    expected_args = 'with these args'
    assert expected_args == testbot.exec_command(cmd)
    assert expected_args == testbot.exec_command(cmd_underscore)
    cmd = '!run lots of subcommands with these args'
    cmd_underscore = '!run_lots_of_subcommands with these args'
    assert expected_args == testbot.exec_command(cmd)
    assert expected_args == testbot.exec_command(cmd_underscore)

def test_command_not_found_with_space_in_bot_prefix(testbot):
    if False:
        i = 10
        return i + 15
    testbot.bot_config.BOT_PREFIX = '! '
    assert 'Command "blah" not found.' in testbot.exec_command('! blah')
    assert 'Command "blah" / "blah toto" not found.' in testbot.exec_command('! blah toto')

def test_mock_injection(testbot):
    if False:
        for i in range(10):
            print('nop')
    helper_mock = MagicMock()
    helper_mock.return_value = 'foo'
    mock_dict = {'helper_method': helper_mock}
    testbot.inject_mocks('Dummy', mock_dict)
    assert 'foo' in testbot.exec_command('!baz')

def test_multiline_command(testbot):
    if False:
        while True:
            i = 10
    testbot.assertInCommand('\n        !bar title\n        first line of body\n        second line of body\n        ', '!bar title\nfirst line of body\nsecond line of body', dedent=True)

def test_plugin_info_command(testbot):
    if False:
        while True:
            i = 10
    output = testbot.exec_command('!plugin info Help')
    assert 'name: Help' in output
    assert 'module: help' in output
    assert 'help.py' in output
    assert 'log level: NOTSET' in output