from os import path
import pytest
extra_plugin_dir = path.join(path.dirname(path.realpath(__file__)), 'test_link')

def test_linked_plugin_here(testbot):
    if False:
        for i in range(10):
            print('nop')
    testbot.push_message('!status plugins')
    assert 'Dummy' in testbot.pop_message()