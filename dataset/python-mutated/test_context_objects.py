from __future__ import annotations
import argparse
import pytest
from ansible.module_utils.common.collections import ImmutableDict
from ansible.utils import context_objects as co
MAKE_IMMUTABLE_DATA = ((u'くらとみ', u'くらとみ'), (42, 42), ({u'café': u'くらとみ'}, ImmutableDict({u'café': u'くらとみ'})), ([1, u'café', u'くらとみ'], (1, u'café', u'くらとみ')), (set((1, u'café', u'くらとみ')), frozenset((1, u'café', u'くらとみ'))), ({u'café': [1, set(u'ñ')]}, ImmutableDict({u'café': (1, frozenset(u'ñ'))})), ([set((1, 2)), {u'くらとみ': 3}], (frozenset((1, 2)), ImmutableDict({u'くらとみ': 3}))))

@pytest.mark.parametrize('data, expected', MAKE_IMMUTABLE_DATA)
def test_make_immutable(data, expected):
    if False:
        for i in range(10):
            print('nop')
    assert co._make_immutable(data) == expected

def test_cliargs_from_dict():
    if False:
        for i in range(10):
            print('nop')
    old_dict = {'tags': [u'production', u'webservers'], 'check_mode': True, 'start_at_task': u'Start with くらとみ'}
    expected = frozenset((('tags', (u'production', u'webservers')), ('check_mode', True), ('start_at_task', u'Start with くらとみ')))
    assert frozenset(co.CLIArgs(old_dict).items()) == expected

def test_cliargs():
    if False:
        for i in range(10):
            print('nop')

    class FakeOptions:
        pass
    options = FakeOptions()
    options.tags = [u'production', u'webservers']
    options.check_mode = True
    options.start_at_task = u'Start with くらとみ'
    expected = frozenset((('tags', (u'production', u'webservers')), ('check_mode', True), ('start_at_task', u'Start with くらとみ')))
    assert frozenset(co.CLIArgs.from_options(options).items()) == expected

def test_cliargs_argparse():
    if False:
        return 10
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('integers', metavar='N', type=int, nargs='+', help='an integer for the accumulator')
    parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, default=max, help='sum the integers (default: find the max)')
    args = parser.parse_args([u'--sum', u'1', u'2'])
    expected = frozenset((('accumulate', sum), ('integers', (1, 2))))
    assert frozenset(co.CLIArgs.from_options(args).items()) == expected