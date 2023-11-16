import argparse
import re
import unittest
from unittest.mock import patch, Mock
import argcomplete
import coalib.parsing.DefaultArgParser
from coalib.collecting.Collectors import get_all_bears_names
from coalib.parsing.DefaultArgParser import CustomFormatter, default_arg_parser

def _get_arg(parser, arg):
    if False:
        for i in range(10):
            print('nop')
    actions = parser.__dict__['_action_groups'][0].__dict__['_actions']
    args = [item for item in actions if arg in item.option_strings]
    return args[0]

class CustomFormatterTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        arg_parser = argparse.ArgumentParser(formatter_class=CustomFormatter)
        arg_parser.add_argument('-a', '--all', nargs='?', const=True, metavar='BOOL')
        arg_parser.add_argument('TARGETS', nargs='*')
        self.output = arg_parser.format_help()

    def test_metavar_in_usage(self):
        if False:
            while True:
                i = 10
        match = re.search('usage:.+(-a \\[BOOL\\]).+\\n\\n', self.output, flags=re.DOTALL)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), '-a [BOOL]')

    def test_metavar_not_in_optional_args_sections(self):
        if False:
            for i in range(10):
                print('nop')
        match = re.search('optional arguments:.+(-a, --all).*', self.output, flags=re.DOTALL)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), '-a, --all')

class AutocompleteTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self._old_argcomplete = coalib.parsing.DefaultArgParser.argcomplete

    def tearDown(self):
        if False:
            while True:
                i = 10
        coalib.parsing.DefaultArgParser.argcomplete = self._old_argcomplete

    def test_argcomplete_missing(self):
        if False:
            return 10
        if coalib.parsing.DefaultArgParser.argcomplete is not None:
            coalib.parsing.DefaultArgParser.argcomplete = None
        real_importer = __import__

        def import_if_not_argcomplete(arg, *args, **kw):
            if False:
                while True:
                    i = 10
            if arg == 'argcomplete':
                raise ImportError('import missing: %s' % arg)
            else:
                return real_importer(arg, *args, **kw)
        mock = Mock(side_effect=import_if_not_argcomplete)
        with patch('builtins.__import__', new=mock):
            default_arg_parser()
        self.assertFalse(coalib.parsing.DefaultArgParser.argcomplete)

    def test_argcomplete_imported(self):
        if False:
            for i in range(10):
                print('nop')
        if coalib.parsing.DefaultArgParser.argcomplete is not None:
            coalib.parsing.DefaultArgParser.argcomplete = None
        parser = default_arg_parser()
        self.assertEqual(coalib.parsing.DefaultArgParser.argcomplete, argcomplete)
        arg = _get_arg(parser, '--bears')
        self.assertTrue(hasattr(arg, 'completer'))
        bears = list(arg.completer())
        self.assertEqual(bears, get_all_bears_names())

    def test_argcomplete_missing_other(self):
        if False:
            while True:
                i = 10
        if coalib.parsing.DefaultArgParser.argcomplete is not None:
            coalib.parsing.DefaultArgParser.argcomplete = None
        real_importer = __import__

        def import_if_not_bear_names(arg, *args, **kw):
            if False:
                return 10
            if arg == 'coalib.collecting.Collectors':
                raise ImportError('import missing: %s' % arg)
            else:
                return real_importer(arg, *args, **kw)
        mock = Mock(side_effect=import_if_not_bear_names)
        with patch('builtins.__import__', new=mock):
            parser = default_arg_parser()
        self.assertTrue(coalib.parsing.DefaultArgParser.argcomplete)
        arg = _get_arg(parser, '--bears')
        self.assertFalse(hasattr(arg, 'completer'))