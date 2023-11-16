import pytest
from ...constants import *
from ...helpers.nanorst import RstToTextLazy, rst_to_terminal
from . import Archiver, cmd

def get_all_parsers():
    if False:
        return 10
    parser = Archiver(prog='borg').build_parser()
    borgfs_parser = Archiver(prog='borgfs').build_parser()
    parsers = {}

    def discover_level(prefix, parser, Archiver, extra_choices=None):
        if False:
            i = 10
            return i + 15
        choices = {}
        for action in parser._actions:
            if action.choices is not None and 'SubParsersAction' in str(action.__class__):
                for (command, parser) in action.choices.items():
                    choices[prefix + command] = parser
        if extra_choices is not None:
            choices.update(extra_choices)
        if prefix and (not choices):
            return
        for (command, parser) in sorted(choices.items()):
            discover_level(command + ' ', parser, Archiver)
            parsers[command] = parser
    discover_level('', parser, Archiver, {'borgfs': borgfs_parser})
    return parsers

def test_usage(archiver):
    if False:
        for i in range(10):
            print('nop')
    cmd(archiver)
    cmd(archiver, '-h')

def test_help(archiver):
    if False:
        i = 10
        return i + 15
    assert 'Borg' in cmd(archiver, 'help')
    assert 'patterns' in cmd(archiver, 'help', 'patterns')
    assert 'creates a new, empty repository' in cmd(archiver, 'help', 'rcreate')
    assert 'positional arguments' not in cmd(archiver, 'help', 'rcreate', '--epilog-only')
    assert 'creates a new, empty repository' not in cmd(archiver, 'help', 'rcreate', '--usage-only')

@pytest.mark.parametrize('command, parser', list(get_all_parsers().items()))
def test_help_formatting(command, parser):
    if False:
        while True:
            i = 10
    if isinstance(parser.epilog, RstToTextLazy):
        assert parser.epilog.rst

@pytest.mark.parametrize('topic', list(Archiver.helptext.keys()))
def test_help_formatting_helptexts(topic):
    if False:
        i = 10
        return i + 15
    helptext = Archiver.helptext[topic]
    assert str(rst_to_terminal(helptext))