import pytest
from argparse import ArgumentParser
from unittest.mock import Mock
from httpie.cli.utils import LazyChoices

def test_lazy_choices():
    if False:
        while True:
            i = 10
    mock = Mock()
    getter = mock.getter
    getter.return_value = ['a', 'b', 'c']
    parser = ArgumentParser()
    parser.register('action', 'lazy_choices', LazyChoices)
    parser.add_argument('--option', help='the regular option', default='a', metavar='SYMBOL', choices=['a', 'b'])
    parser.add_argument('--lazy-option', help='the lazy option', default='a', metavar='SYMBOL', action='lazy_choices', getter=getter, cache=False)
    getter.assert_not_called()
    parser.parse_args([])
    getter.assert_not_called()
    parser.parse_args(['--option', 'b'])
    getter.assert_not_called()
    parser.parse_args(['--lazy-option', 'c'])
    getter.assert_called()
    getter.reset_mock()
    with pytest.raises(SystemExit):
        parser.parse_args(['--lazy-option', 'z'])
    getter.assert_called()
    getter.reset_mock()

def test_lazy_choices_help():
    if False:
        while True:
            i = 10
    mock = Mock()
    getter = mock.getter
    getter.return_value = ['a', 'b', 'c']
    help_formatter = mock.help_formatter
    help_formatter.return_value = '<my help>'
    parser = ArgumentParser()
    parser.register('action', 'lazy_choices', LazyChoices)
    parser.add_argument('--lazy-option', default='a', metavar='SYMBOL', action='lazy_choices', getter=getter, help_formatter=help_formatter, cache=False)
    getter.assert_not_called()
    parser.parse_args([])
    getter.assert_not_called()
    help_formatter.assert_not_called()
    parser.parse_args(['--lazy-option', 'b'])
    help_formatter.assert_not_called()
    with pytest.raises(SystemExit):
        parser.parse_args(['--help'])
    help_formatter.assert_called_once_with(['a', 'b', 'c'], isolation_mode=False)