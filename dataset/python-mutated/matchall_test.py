from os import path
import pytest
extra_plugin_dir = path.join(path.dirname(path.realpath(__file__)), 'matchall_plugin')

def test_botmatch_correct(testbot):
    if False:
        for i in range(10):
            print('nop')
    assert 'Works!' in testbot.exec_command('hi hi hi')

def test_botmatch(testbot):
    if False:
        print('Hello World!')
    assert 'Works!' in testbot.exec_command('123123')