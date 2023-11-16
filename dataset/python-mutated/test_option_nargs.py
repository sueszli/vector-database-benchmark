from unittest import TestCase
from unittest.mock import MagicMock
from samcli.commands._utils.custom_options.option_nargs import OptionNargs

class MockRArgs:

    def __init__(self, rargs):
        if False:
            i = 10
            return i + 15
        self.rargs = rargs

class TestOptionNargs(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.name = 'test'
        self.opt = '--use'
        self.prefixes = ['--', '-']
        self.arg = 'first'
        self.rargs_list = ['second', 'third', '--nextopt']
        self.expected_args = tuple([self.arg] + self.rargs_list[:-1])
        self.option_nargs = OptionNargs(param_decls=(self.name, self.opt))

    def test_option(self):
        if False:
            for i in range(10):
                print('nop')
        parser = MagicMock()
        ctx = MagicMock()
        self.option_nargs.add_to_parser(parser=parser, ctx=ctx)
        parser._long_opt.get.assert_called_with(self.opt)
        self.assertEqual(self.option_nargs._nargs_parser, parser._long_opt.get())
        self.option_nargs._nargs_parser.prefixes = self.prefixes
        state = MockRArgs(self.rargs_list)
        parser._long_opt.get().process(self.arg, state)
        self.option_nargs._previous_parser_process.assert_called_with(self.expected_args, state)