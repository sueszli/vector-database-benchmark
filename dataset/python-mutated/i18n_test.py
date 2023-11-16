from os import path
extra_plugin_dir = path.join(path.dirname(path.realpath(__file__)), 'i18n_plugin')

def test_i18n_return(testbot):
    if False:
        return 10
    assert 'язы́к' in testbot.exec_command('!i18n 1')

def test_i18n_simple_name(testbot):
    if False:
        print('Hello World!')
    assert 'OK' in testbot.exec_command('!ру́сский')

def test_i18n_prefix(testbot):
    if False:
        for i in range(10):
            print('nop')
    assert 'OK' in testbot.exec_command('!prefix_ру́сский')
    assert 'OK' in testbot.exec_command('!prefix ру́сский')

def test_i18n_suffix(testbot):
    if False:
        while True:
            i = 10
    assert 'OK' in testbot.exec_command('!ру́сский_suffix')
    assert 'OK' in testbot.exec_command('!ру́сский suffix')