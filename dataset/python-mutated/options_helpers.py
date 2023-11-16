"""Provides helper classes for testing option handling in pip
"""
from optparse import Values
from typing import List, Tuple
from pip._internal.cli import cmdoptions
from pip._internal.cli.base_command import Command
from pip._internal.commands import CommandInfo, commands_dict

class FakeCommand(Command):

    def main(self, args: List[str]) -> Tuple[Values, List[str]]:
        if False:
            while True:
                i = 10
        index_opts = cmdoptions.make_option_group(cmdoptions.index_group, self.parser)
        self.parser.add_option_group(index_opts)
        return self.parse_args(args)

class AddFakeCommandMixin:

    def setup_method(self) -> None:
        if False:
            return 10
        commands_dict['fake'] = CommandInfo('tests.lib.options_helpers', 'FakeCommand', 'fake summary')

    def teardown_method(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        commands_dict.pop('fake')