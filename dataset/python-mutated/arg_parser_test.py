"""Tests for arg_parser."""
import argparse
import sys
from pytype import config as pytype_config
from pytype import datatypes
from pytype.tools import arg_parser
import unittest

def make_parser():
    if False:
        print('Hello World!')
    'Construct a parser to run tests against.'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', type=str, action='store', default='')
    wrapper = datatypes.ParserWrapper(parser)
    wrapper.add_argument('input', nargs='*')
    wrapper.add_argument('-v', '--verbosity', dest='verbosity', type=int, action='store', default=1)
    pytype_config.add_basic_options(wrapper)
    return arg_parser.Parser(parser, pytype_single_args=wrapper.actions)

class TestParser(unittest.TestCase):
    """Test arg_parser.Parser."""

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        super().setUpClass()
        cls.parser = make_parser()

    def test_verbosity(self):
        if False:
            for i in range(10):
                print('nop')
        args = self.parser.parse_args(['--verbosity', '0'])
        self.assertEqual(args.pytype_opts.verbosity, 0)
        args = self.parser.parse_args(['-v1'])
        self.assertEqual(args.pytype_opts.verbosity, 1)

    def test_tool_and_ptype_args(self):
        if False:
            print('Hello World!')
        args = self.parser.parse_args(['--config=test.cfg', '-v1'])
        self.assertEqual(args.tool_args.config, 'test.cfg')
        args = self.parser.parse_args(['-v1'])
        self.assertEqual(args.pytype_opts.verbosity, 1)

    def test_pytype_single_args(self):
        if False:
            i = 10
            return i + 15
        args = self.parser.parse_args(['--disable=import-error'])
        self.assertSequenceEqual(args.pytype_opts.disable, ['import-error'])

    def test_input_file(self):
        if False:
            return 10
        args = self.parser.parse_args(['-v1', 'foo.py'])
        self.assertEqual(args.pytype_opts.verbosity, 1)
        self.assertEqual(args.pytype_opts.input, 'foo.py')

    def test_process_parsed_args(self):
        if False:
            print('Hello World!')
        incoming_args = argparse.Namespace()
        incoming_args.config = 'test.cfg'
        incoming_args.verbosity = 1
        args = self.parser.process_parsed_args(incoming_args)
        self.assertEqual(args.tool_args.config, 'test.cfg')
        self.assertEqual(args.pytype_opts.verbosity, 1)

    def test_override(self):
        if False:
            while True:
                i = 10
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', dest='config', type=str, action='store', default='')
        wrapper = datatypes.ParserWrapper(parser)
        pytype_config.add_basic_options(wrapper)
        parser = arg_parser.Parser(parser, pytype_single_args=wrapper.actions, overrides=['platform'])
        args = parser.parse_args(['--platform', 'plan9', '--disable', 'foo,bar'])
        self.assertEqual(args.tool_args.platform, 'plan9')
        self.assertEqual(args.pytype_opts.platform, sys.platform)
        self.assertEqual(args.pytype_opts.disable, ['foo', 'bar'])
if __name__ == '__main__':
    unittest.main()