"""Style config tests for yapf.reformatter."""
import textwrap
import unittest
from yapf.yapflib import reformatter
from yapf.yapflib import style
from yapftests import yapf_test_helper

class TestsForStyleConfig(yapf_test_helper.YAPFTest):

    def setUp(self):
        if False:
            return 10
        self.current_style = style.DEFAULT_STYLE

    def testSetGlobalStyle(self):
        if False:
            while True:
                i = 10
        try:
            style.SetGlobalStyle(style.CreateYapfStyle())
            unformatted_code = textwrap.dedent("          for i in range(5):\n           print('bar')\n      ")
            expected_formatted_code = textwrap.dedent("          for i in range(5):\n            print('bar')\n      ")
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreatePEP8Style())
            style.DEFAULT_STYLE = self.current_style
        unformatted_code = textwrap.dedent("        for i in range(5):\n         print('bar')\n    ")
        expected_formatted_code = textwrap.dedent("        for i in range(5):\n            print('bar')\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testOperatorNoSpaceStyle(self):
        if False:
            return 10
        try:
            sympy_style = style.CreatePEP8Style()
            sympy_style['NO_SPACES_AROUND_SELECTED_BINARY_OPERATORS'] = style._StringSetConverter('*,/')
            style.SetGlobalStyle(sympy_style)
            unformatted_code = textwrap.dedent("          a = 1+2 * 3 - 4 / 5\n          b = '0' * 1\n      ")
            expected_formatted_code = textwrap.dedent("          a = 1 + 2*3 - 4/5\n          b = '0'*1\n      ")
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreatePEP8Style())
            style.DEFAULT_STYLE = self.current_style

    def testOperatorPrecedenceStyle(self):
        if False:
            return 10
        try:
            pep8_with_precedence = style.CreatePEP8Style()
            pep8_with_precedence['ARITHMETIC_PRECEDENCE_INDICATION'] = True
            style.SetGlobalStyle(pep8_with_precedence)
            unformatted_code = textwrap.dedent('          1+2\n          (1 + 2) * (3 - (4 / 5))\n          a = 1 * 2 + 3 / 4\n          b = 1 / 2 - 3 * 4\n          c = (1 + 2) * (3 - 4)\n          d = (1 - 2) / (3 + 4)\n          e = 1 * 2 - 3\n          f = 1 + 2 + 3 + 4\n          g = 1 * 2 * 3 * 4\n          h = 1 + 2 - 3 + 4\n          i = 1 * 2 / 3 * 4\n          j = (1 * 2 - 3) + 4\n          k = (1 * 2 * 3) + (4 * 5 * 6 * 7 * 8)\n      ')
            expected_formatted_code = textwrap.dedent('          1 + 2\n          (1+2) * (3 - (4/5))\n          a = 1*2 + 3/4\n          b = 1/2 - 3*4\n          c = (1+2) * (3-4)\n          d = (1-2) / (3+4)\n          e = 1*2 - 3\n          f = 1 + 2 + 3 + 4\n          g = 1 * 2 * 3 * 4\n          h = 1 + 2 - 3 + 4\n          i = 1 * 2 / 3 * 4\n          j = (1*2 - 3) + 4\n          k = (1*2*3) + (4*5*6*7*8)\n      ')
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreatePEP8Style())
            style.DEFAULT_STYLE = self.current_style

    def testNoSplitBeforeFirstArgumentStyle1(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            pep8_no_split_before_first = style.CreatePEP8Style()
            pep8_no_split_before_first['SPLIT_BEFORE_FIRST_ARGUMENT'] = False
            pep8_no_split_before_first['SPLIT_BEFORE_NAMED_ASSIGNS'] = False
            style.SetGlobalStyle(pep8_no_split_before_first)
            formatted_code = textwrap.dedent('          # Example from in-code MustSplit comments\n          foo = outer_function_call(fitting_inner_function_call(inner_arg1, inner_arg2),\n                                    outer_arg1, outer_arg2)\n\n          foo = outer_function_call(\n              not_fitting_inner_function_call(inner_arg1, inner_arg2), outer_arg1,\n              outer_arg2)\n\n          # Examples Issue#424\n          a_super_long_version_of_print(argument1, argument2, argument3, argument4,\n                                        argument5, argument6, argument7)\n\n          CREDS_FILE = os.path.join(os.path.expanduser(\'~\'),\n                                    \'apis/super-secret-admin-creds.json\')\n\n          # Examples Issue#556\n          i_take_a_lot_of_params(arg1, param1=very_long_expression1(),\n                                 param2=very_long_expression2(),\n                                 param3=very_long_expression3(),\n                                 param4=very_long_expression4())\n\n          # Examples Issue#590\n          plt.plot(numpy.linspace(0, 1, 10), numpy.linspace(0, 1, 10), marker="x",\n                   color="r")\n\n          plt.plot(veryverylongvariablename, veryverylongvariablename, marker="x",\n                   color="r")\n      ')
            llines = yapf_test_helper.ParseAndUnwrap(formatted_code)
            self.assertCodeEqual(formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreatePEP8Style())
            style.DEFAULT_STYLE = self.current_style

    def testNoSplitBeforeFirstArgumentStyle2(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            pep8_no_split_before_first = style.CreatePEP8Style()
            pep8_no_split_before_first['SPLIT_BEFORE_FIRST_ARGUMENT'] = False
            pep8_no_split_before_first['SPLIT_BEFORE_NAMED_ASSIGNS'] = True
            style.SetGlobalStyle(pep8_no_split_before_first)
            formatted_code = textwrap.dedent('          # Examples Issue#556\n          i_take_a_lot_of_params(arg1,\n                                 param1=very_long_expression1(),\n                                 param2=very_long_expression2(),\n                                 param3=very_long_expression3(),\n                                 param4=very_long_expression4())\n\n          # Examples Issue#590\n          plt.plot(numpy.linspace(0, 1, 10),\n                   numpy.linspace(0, 1, 10),\n                   marker="x",\n                   color="r")\n\n          plt.plot(veryverylongvariablename,\n                   veryverylongvariablename,\n                   marker="x",\n                   color="r")\n      ')
            llines = yapf_test_helper.ParseAndUnwrap(formatted_code)
            self.assertCodeEqual(formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreatePEP8Style())
            style.DEFAULT_STYLE = self.current_style
if __name__ == '__main__':
    unittest.main()