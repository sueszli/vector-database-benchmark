"""Tests for fire docstrings module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import docstrings
from fire import testutils
DocstringInfo = docstrings.DocstringInfo
ArgInfo = docstrings.ArgInfo
KwargInfo = docstrings.KwargInfo

class DocstringsTest(testutils.BaseTestCase):

    def test_one_line_simple(self):
        if False:
            for i in range(10):
                print('nop')
        docstring = 'A simple one line docstring.'
        docstring_info = docstrings.parse(docstring)
        expected_docstring_info = DocstringInfo(summary='A simple one line docstring.')
        self.assertEqual(expected_docstring_info, docstring_info)

    def test_one_line_simple_whitespace(self):
        if False:
            return 10
        docstring = '\n      A simple one line docstring.\n    '
        docstring_info = docstrings.parse(docstring)
        expected_docstring_info = DocstringInfo(summary='A simple one line docstring.')
        self.assertEqual(expected_docstring_info, docstring_info)

    def test_one_line_too_long(self):
        if False:
            return 10
        docstring = 'A one line docstring that is both a little too verbose and a little too long so it keeps going well beyond a reasonable length for a one-liner.\n    '
        docstring_info = docstrings.parse(docstring)
        expected_docstring_info = DocstringInfo(summary='A one line docstring that is both a little too verbose and a little too long so it keeps going well beyond a reasonable length for a one-liner.')
        self.assertEqual(expected_docstring_info, docstring_info)

    def test_one_line_runs_over(self):
        if False:
            i = 10
            return i + 15
        docstring = 'A one line docstring that is both a little too verbose and a little too long\n    so it runs onto a second line.\n    '
        docstring_info = docstrings.parse(docstring)
        expected_docstring_info = DocstringInfo(summary='A one line docstring that is both a little too verbose and a little too long so it runs onto a second line.')
        self.assertEqual(expected_docstring_info, docstring_info)

    def test_one_line_runs_over_whitespace(self):
        if False:
            return 10
        docstring = '\n      A one line docstring that is both a little too verbose and a little too long\n      so it runs onto a second line.\n    '
        docstring_info = docstrings.parse(docstring)
        expected_docstring_info = DocstringInfo(summary='A one line docstring that is both a little too verbose and a little too long so it runs onto a second line.')
        self.assertEqual(expected_docstring_info, docstring_info)

    def test_google_format_args_only(self):
        if False:
            return 10
        docstring = 'One line description.\n\n    Args:\n      arg1: arg1_description\n      arg2: arg2_description\n    '
        docstring_info = docstrings.parse(docstring)
        expected_docstring_info = DocstringInfo(summary='One line description.', args=[ArgInfo(name='arg1', description='arg1_description'), ArgInfo(name='arg2', description='arg2_description')])
        self.assertEqual(expected_docstring_info, docstring_info)

    def test_google_format_arg_named_args(self):
        if False:
            return 10
        docstring = '\n    Args:\n      args: arg_description\n    '
        docstring_info = docstrings.parse(docstring)
        expected_docstring_info = DocstringInfo(args=[ArgInfo(name='args', description='arg_description')])
        self.assertEqual(expected_docstring_info, docstring_info)

    def test_google_format_typed_args_and_returns(self):
        if False:
            i = 10
            return i + 15
        docstring = 'Docstring summary.\n\n    This is a longer description of the docstring. It spans multiple lines, as\n    is allowed.\n\n    Args:\n        param1 (int): The first parameter.\n        param2 (str): The second parameter.\n\n    Returns:\n        bool: The return value. True for success, False otherwise.\n    '
        docstring_info = docstrings.parse(docstring)
        expected_docstring_info = DocstringInfo(summary='Docstring summary.', description='This is a longer description of the docstring. It spans multiple lines, as\nis allowed.', args=[ArgInfo(name='param1', type='int', description='The first parameter.'), ArgInfo(name='param2', type='str', description='The second parameter.')], returns='bool: The return value. True for success, False otherwise.')
        self.assertEqual(expected_docstring_info, docstring_info)

    def test_google_format_multiline_arg_description(self):
        if False:
            print('Hello World!')
        docstring = 'Docstring summary.\n\n    This is a longer description of the docstring. It spans multiple lines, as\n    is allowed.\n\n    Args:\n        param1 (int): The first parameter.\n        param2 (str): The second parameter. This has a lot of text, enough to\n        cover two lines.\n    '
        docstring_info = docstrings.parse(docstring)
        expected_docstring_info = DocstringInfo(summary='Docstring summary.', description='This is a longer description of the docstring. It spans multiple lines, as\nis allowed.', args=[ArgInfo(name='param1', type='int', description='The first parameter.'), ArgInfo(name='param2', type='str', description='The second parameter. This has a lot of text, enough to cover two lines.')])
        self.assertEqual(expected_docstring_info, docstring_info)

    def test_rst_format_typed_args_and_returns(self):
        if False:
            for i in range(10):
                print('nop')
        docstring = 'Docstring summary.\n\n    This is a longer description of the docstring. It spans across multiple\n    lines.\n\n    :param arg1: Description of arg1.\n    :type arg1: str.\n    :param arg2: Description of arg2.\n    :type arg2: bool.\n    :returns:  int -- description of the return value.\n    :raises: AttributeError, KeyError\n    '
        docstring_info = docstrings.parse(docstring)
        expected_docstring_info = DocstringInfo(summary='Docstring summary.', description='This is a longer description of the docstring. It spans across multiple\nlines.', args=[ArgInfo(name='arg1', type='str', description='Description of arg1.'), ArgInfo(name='arg2', type='bool', description='Description of arg2.')], returns='int -- description of the return value.', raises='AttributeError, KeyError')
        self.assertEqual(expected_docstring_info, docstring_info)

    def test_numpy_format_typed_args_and_returns(self):
        if False:
            i = 10
            return i + 15
        docstring = 'Docstring summary.\n\n    This is a longer description of the docstring. It spans across multiple\n    lines.\n\n    Parameters\n    ----------\n    param1 : int\n        The first parameter.\n    param2 : str\n        The second parameter.\n\n    Returns\n    -------\n    bool\n        True if successful, False otherwise.\n    '
        docstring_info = docstrings.parse(docstring)
        expected_docstring_info = DocstringInfo(summary='Docstring summary.', description='This is a longer description of the docstring. It spans across multiple\nlines.', args=[ArgInfo(name='param1', type='int', description='The first parameter.'), ArgInfo(name='param2', type='str', description='The second parameter.')], returns='bool True if successful, False otherwise.')
        self.assertEqual(expected_docstring_info, docstring_info)

    def test_numpy_format_multiline_arg_description(self):
        if False:
            return 10
        docstring = 'Docstring summary.\n\n    This is a longer description of the docstring. It spans across multiple\n    lines.\n\n    Parameters\n    ----------\n    param1 : int\n        The first parameter.\n    param2 : str\n        The second parameter. This has a lot of text, enough to cover two\n        lines.\n    '
        docstring_info = docstrings.parse(docstring)
        expected_docstring_info = DocstringInfo(summary='Docstring summary.', description='This is a longer description of the docstring. It spans across multiple\nlines.', args=[ArgInfo(name='param1', type='int', description='The first parameter.'), ArgInfo(name='param2', type='str', description='The second parameter. This has a lot of text, enough to cover two lines.')])
        self.assertEqual(expected_docstring_info, docstring_info)

    def test_multisection_docstring(self):
        if False:
            while True:
                i = 10
        docstring = 'Docstring summary.\n\n    This is the first section of a docstring description.\n\n    This is the second section of a docstring description. This docstring\n    description has just two sections.\n    '
        docstring_info = docstrings.parse(docstring)
        expected_docstring_info = DocstringInfo(summary='Docstring summary.', description='This is the first section of a docstring description.\n\nThis is the second section of a docstring description. This docstring\ndescription has just two sections.')
        self.assertEqual(expected_docstring_info, docstring_info)

    def test_google_section_with_blank_first_line(self):
        if False:
            while True:
                i = 10
        docstring = 'Inspired by requests HTTPAdapter docstring.\n\n    :param x: Simple param.\n\n    Usage:\n\n      >>> import requests\n    '
        docstring_info = docstrings.parse(docstring)
        self.assertEqual('Inspired by requests HTTPAdapter docstring.', docstring_info.summary)

    def test_ill_formed_docstring(self):
        if False:
            return 10
        docstring = 'Docstring summary.\n\n    args: raises ::\n    :\n    pathological docstrings should not fail, and ideally should behave\n    reasonably.\n    '
        docstrings.parse(docstring)

    def test_strip_blank_lines(self):
        if False:
            print('Hello World!')
        lines = ['   ', '  foo  ', '   ']
        expected_output = ['  foo  ']
        self.assertEqual(expected_output, docstrings._strip_blank_lines(lines))

    def test_numpy_colon_in_description(self):
        if False:
            for i in range(10):
                print('nop')
        docstring = '\n     Greets name.\n\n     Arguments\n     ---------\n     name : str\n         name, default : World\n     arg2 : int\n         arg2, default:None\n     arg3 : bool\n     '
        docstring_info = docstrings.parse(docstring)
        expected_docstring_info = DocstringInfo(summary='Greets name.', description=None, args=[ArgInfo(name='name', type='str', description='name, default : World'), ArgInfo(name='arg2', type='int', description='arg2, default:None'), ArgInfo(name='arg3', type='bool', description=None)])
        self.assertEqual(expected_docstring_info, docstring_info)

    def test_rst_format_typed_args_and_kwargs(self):
        if False:
            print('Hello World!')
        docstring = 'Docstring summary.\n\n    :param arg1: Description of arg1.\n    :type arg1: str.\n    :key arg2: Description of arg2.\n    :type arg2: bool.\n    :key arg3: Description of arg3.\n    :type arg3: str.\n    '
        docstring_info = docstrings.parse(docstring)
        expected_docstring_info = DocstringInfo(summary='Docstring summary.', args=[ArgInfo(name='arg1', type='str', description='Description of arg1.'), KwargInfo(name='arg2', type='bool', description='Description of arg2.'), KwargInfo(name='arg3', type='str', description='Description of arg3.')])
        self.assertEqual(expected_docstring_info, docstring_info)
if __name__ == '__main__':
    testutils.main()