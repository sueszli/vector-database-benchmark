from os import path
extra_plugin_dir = path.join(path.dirname(path.realpath(__file__)), 'multi_plugin')

def test_first(testbot):
    if False:
        return 10
    r = testbot.exec_command('!myname')
    assert 'Multi1' == r or 'Multi2' == r

def test_second(testbot):
    if False:
        return 10
    assert 'Multi2' == testbot.exec_command('!multi2-myname') or 'Multi1' == testbot.exec_command('!multi1-myname')