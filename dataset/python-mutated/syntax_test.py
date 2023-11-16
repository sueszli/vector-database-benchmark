from os import path
extra_plugin_dir = path.join(path.dirname(path.realpath(__file__)), 'syntax_plugin')

def test_nosyntax(testbot):
    if False:
        for i in range(10):
            print('nop')
    assert testbot.bot.commands['foo_nosyntax']._err_command_syntax is None

def test_syntax(testbot):
    if False:
        i = 10
        return i + 15
    assert testbot.bot.commands['foo']._err_command_syntax == '[optional] <mandatory>'

def test_re_syntax(testbot):
    if False:
        while True:
            i = 10
    assert testbot.bot.re_commands['re_foo']._err_command_syntax == '.*'

def test_arg_syntax(testbot):
    if False:
        return 10
    assert testbot.bot.commands['arg_foo']._err_command_syntax == '[-h] [--repeat-count REPEAT] value'