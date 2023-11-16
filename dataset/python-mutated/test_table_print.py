import io
from contextlib import redirect_stdout
from collections import OrderedDict
from unittest import TestCase
from samcli.commands._utils.table_print import pprint_column_names, pprint_columns
TABLE_FORMAT_STRING = '{Alpha:<{0}} {Beta:<{1}} {Gamma:<{2}}'
TABLE_FORMAT_ARGS = OrderedDict({'Alpha': 'Alpha', 'Beta': 'Beta', 'Gamma': 'Gamma'})

class TestTablePrint(TestCase):

    def setUp(self):
        if False:
            return 10
        self.redirect_out = io.StringIO()

    def test_pprint_column_names(self):
        if False:
            print('Hello World!')

        @pprint_column_names(TABLE_FORMAT_STRING, TABLE_FORMAT_ARGS)
        def to_be_decorated(*args, **kwargs):
            if False:
                return 10
            pass
        with redirect_stdout(self.redirect_out):
            to_be_decorated()
        output = '------------------------------------------------------------------------------------------------\nAlpha                            Beta                             Gamma                          \n------------------------------------------------------------------------------------------------\n------------------------------------------------------------------------------------------------\n\n'
        self.assertEqual(output, self.redirect_out.getvalue())

    def test_pprint_column_names_and_text(self):
        if False:
            while True:
                i = 10

        @pprint_column_names(TABLE_FORMAT_STRING, TABLE_FORMAT_ARGS)
        def to_be_decorated(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            pprint_columns(columns=['A', 'B', 'C'], width=kwargs['width'], margin=kwargs['margin'], format_args=kwargs['format_args'], format_string=TABLE_FORMAT_STRING, columns_dict=TABLE_FORMAT_ARGS.copy())
        with redirect_stdout(self.redirect_out):
            to_be_decorated()
        output = '------------------------------------------------------------------------------------------------\nAlpha                            Beta                             Gamma                          \n------------------------------------------------------------------------------------------------\nA                                B                                C                              \n------------------------------------------------------------------------------------------------\n\n'
        self.assertEqual(output, self.redirect_out.getvalue())

    def test_pprint_exceptions_with_no_column_names(self):
        if False:
            return 10
        with self.assertRaises(ValueError):

            @pprint_column_names(TABLE_FORMAT_STRING, {})
            def to_be_decorated(*args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                pprint_columns(columns=['A', 'B', 'C'], width=kwargs['width'], margin=kwargs['margin'], format_args=kwargs['format_args'], format_string=TABLE_FORMAT_STRING, columns_dict=TABLE_FORMAT_ARGS.copy())

    def test_pprint_exceptions_with_too_many_column_names(self):
        if False:
            i = 10
            return i + 15
        massive_dictionary = {str(i): str(i) for i in range(100)}
        with self.assertRaises(ValueError):

            @pprint_column_names(TABLE_FORMAT_STRING, massive_dictionary)
            def to_be_decorated(*args, **kwargs):
                if False:
                    while True:
                        i = 10
                pprint_columns(columns=['A', 'B', 'C'], width=kwargs['width'], margin=kwargs['margin'], format_args=kwargs['format_args'], format_string=TABLE_FORMAT_STRING, columns_dict=TABLE_FORMAT_ARGS.copy())