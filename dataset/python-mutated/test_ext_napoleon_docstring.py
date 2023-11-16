"""Tests for :mod:`sphinx.ext.napoleon.docstring` module."""
import re
from collections import namedtuple
from inspect import cleandoc
from textwrap import dedent
from unittest import mock
import pytest
from sphinx.ext.napoleon import Config
from sphinx.ext.napoleon.docstring import GoogleDocstring, NumpyDocstring, _convert_numpy_type_spec, _recombine_set_tokens, _token_type, _tokenize_type_spec
from .ext_napoleon_pep526_data_google import PEP526GoogleClass
from .ext_napoleon_pep526_data_numpy import PEP526NumpyClass

class NamedtupleSubclass(namedtuple('NamedtupleSubclass', ('attr1', 'attr2'))):
    """Sample namedtuple subclass

    Attributes
    ----------
    attr1 : Arbitrary type
        Quick description of attr1
    attr2 : Another arbitrary type
        Quick description of attr2
    attr3 : Type

        Adds a newline after the type

    """
    __slots__ = ()

    def __new__(cls, attr1, attr2=None):
        if False:
            for i in range(10):
                print('nop')
        return super().__new__(cls, attr1, attr2)

class TestNamedtupleSubclass:

    def test_attributes_docstring(self):
        if False:
            for i in range(10):
                print('nop')
        config = Config()
        actual = str(NumpyDocstring(cleandoc(NamedtupleSubclass.__doc__), config=config, app=None, what='class', name='NamedtupleSubclass', obj=NamedtupleSubclass))
        expected = 'Sample namedtuple subclass\n\n.. attribute:: attr1\n\n   Quick description of attr1\n\n   :type: Arbitrary type\n\n.. attribute:: attr2\n\n   Quick description of attr2\n\n   :type: Another arbitrary type\n\n.. attribute:: attr3\n\n   Adds a newline after the type\n\n   :type: Type\n'
        assert expected == actual

class TestInlineAttribute:
    inline_google_docstring = 'inline description with ``a : in code``, a :ref:`reference`, a `link <https://foo.bar>`_, a :meta public:, a :meta field: value and an host:port and HH:MM strings.'

    @staticmethod
    def _docstring(source):
        if False:
            while True:
                i = 10
        rst = GoogleDocstring(source, config=Config(), app=None, what='attribute', name='some_data', obj=0)
        return str(rst)

    def test_class_data_member(self):
        if False:
            i = 10
            return i + 15
        source = 'data member description:\n\n- a: b'
        actual = self._docstring(source).splitlines()
        assert actual == ['data member description:', '', '- a: b']

    def test_class_data_member_inline(self):
        if False:
            i = 10
            return i + 15
        source = f'CustomType: {self.inline_google_docstring}'
        actual = self._docstring(source).splitlines()
        assert actual == [self.inline_google_docstring, '', ':type: CustomType']

    def test_class_data_member_inline_no_type(self):
        if False:
            return 10
        source = self.inline_google_docstring
        actual = self._docstring(source).splitlines()
        assert actual == [source]

    def test_class_data_member_inline_ref_in_type(self):
        if False:
            print('Hello World!')
        source = f':class:`int`: {self.inline_google_docstring}'
        actual = self._docstring(source).splitlines()
        assert actual == [self.inline_google_docstring, '', ':type: :class:`int`']

class TestGoogleDocstring:
    docstrings = [('Single line summary', 'Single line summary'), ('\n        Single line summary\n\n        Extended description\n\n        ', '\n        Single line summary\n\n        Extended description\n        '), ('\n        Single line summary\n\n        Args:\n          arg1(str):Extended\n            description of arg1\n        ', '\n        Single line summary\n\n        :Parameters: **arg1** (*str*) -- Extended\n                     description of arg1\n        '), ('\n        Single line summary\n\n        Args:\n          arg1(str):Extended\n            description of arg1\n          arg2 ( int ) : Extended\n            description of arg2\n\n        Keyword Args:\n          kwarg1(str):Extended\n            description of kwarg1\n          kwarg2 ( int ) : Extended\n            description of kwarg2', '\n        Single line summary\n\n        :Parameters: * **arg1** (*str*) -- Extended\n                       description of arg1\n                     * **arg2** (*int*) -- Extended\n                       description of arg2\n\n        :Keyword Arguments: * **kwarg1** (*str*) -- Extended\n                              description of kwarg1\n                            * **kwarg2** (*int*) -- Extended\n                              description of kwarg2\n        '), ('\n        Single line summary\n\n        Arguments:\n          arg1(str):Extended\n            description of arg1\n          arg2 ( int ) : Extended\n            description of arg2\n\n        Keyword Arguments:\n          kwarg1(str):Extended\n            description of kwarg1\n          kwarg2 ( int ) : Extended\n            description of kwarg2', '\n        Single line summary\n\n        :Parameters: * **arg1** (*str*) -- Extended\n                       description of arg1\n                     * **arg2** (*int*) -- Extended\n                       description of arg2\n\n        :Keyword Arguments: * **kwarg1** (*str*) -- Extended\n                              description of kwarg1\n                            * **kwarg2** (*int*) -- Extended\n                              description of kwarg2\n        '), ('\n        Single line summary\n\n        Return:\n          str:Extended\n          description of return value\n        ', '\n        Single line summary\n\n        :returns: *str* -- Extended\n                  description of return value\n        '), ('\n        Single line summary\n\n        Returns:\n          str:Extended\n          description of return value\n        ', '\n        Single line summary\n\n        :returns: *str* -- Extended\n                  description of return value\n        '), ('\n        Single line summary\n\n        Returns:\n          Extended\n          description of return value\n        ', '\n        Single line summary\n\n        :returns: Extended\n                  description of return value\n        '), ('\n        Single line summary\n\n        Returns:\n          Extended\n        ', '\n        Single line summary\n\n        :returns: Extended\n        '), ('\n        Single line summary\n\n        Args:\n          arg1(str):Extended\n            description of arg1\n          *args: Variable length argument list.\n          **kwargs: Arbitrary keyword arguments.\n        ', '\n        Single line summary\n\n        :Parameters: * **arg1** (*str*) -- Extended\n                       description of arg1\n                     * **\\*args** -- Variable length argument list.\n                     * **\\*\\*kwargs** -- Arbitrary keyword arguments.\n        '), ('\n        Single line summary\n\n        Args:\n          arg1 (list(int)): Description\n          arg2 (list[int]): Description\n          arg3 (dict(str, int)): Description\n          arg4 (dict[str, int]): Description\n        ', '\n        Single line summary\n\n        :Parameters: * **arg1** (*list(int)*) -- Description\n                     * **arg2** (*list[int]*) -- Description\n                     * **arg3** (*dict(str, int)*) -- Description\n                     * **arg4** (*dict[str, int]*) -- Description\n        '), ('\n        Single line summary\n\n        Receive:\n          arg1 (list(int)): Description\n          arg2 (list[int]): Description\n        ', '\n        Single line summary\n\n        :Receives: * **arg1** (*list(int)*) -- Description\n                   * **arg2** (*list[int]*) -- Description\n        '), ('\n        Single line summary\n\n        Receives:\n          arg1 (list(int)): Description\n          arg2 (list[int]): Description\n        ', '\n        Single line summary\n\n        :Receives: * **arg1** (*list(int)*) -- Description\n                   * **arg2** (*list[int]*) -- Description\n        '), ('\n        Single line summary\n\n        Yield:\n          str:Extended\n          description of yielded value\n        ', '\n        Single line summary\n\n        :Yields: *str* -- Extended\n                 description of yielded value\n        '), ('\n        Single line summary\n\n        Yields:\n          Extended\n          description of yielded value\n        ', '\n        Single line summary\n\n        :Yields: Extended\n                 description of yielded value\n        '), ('\n        Single line summary\n\n        Args:\n\n          arg1 (list of str): Extended\n              description of arg1.\n          arg2 (tuple of int): Extended\n              description of arg2.\n          arg3 (tuple of list of float): Extended\n              description of arg3.\n          arg4 (int, float, or list of bool): Extended\n              description of arg4.\n          arg5 (list of int, float, or bool): Extended\n              description of arg5.\n          arg6 (list of int or float): Extended\n              description of arg6.\n        ', '\n        Single line summary\n\n        :Parameters: * **arg1** (*list of str*) -- Extended\n                       description of arg1.\n                     * **arg2** (*tuple of int*) -- Extended\n                       description of arg2.\n                     * **arg3** (*tuple of list of float*) -- Extended\n                       description of arg3.\n                     * **arg4** (*int, float, or list of bool*) -- Extended\n                       description of arg4.\n                     * **arg5** (*list of int, float, or bool*) -- Extended\n                       description of arg5.\n                     * **arg6** (*list of int or float*) -- Extended\n                       description of arg6.\n        ')]

    def test_sphinx_admonitions(self):
        if False:
            print('Hello World!')
        admonition_map = {'Attention': 'attention', 'Caution': 'caution', 'Danger': 'danger', 'Error': 'error', 'Hint': 'hint', 'Important': 'important', 'Note': 'note', 'Tip': 'tip', 'Todo': 'todo', 'Warning': 'warning', 'Warnings': 'warning'}
        config = Config()
        for (section, admonition) in admonition_map.items():
            actual = str(GoogleDocstring(f'{section}:\n    this is the first line\n\n    and this is the second line\n', config))
            expect = f'.. {admonition}::\n\n   this is the first line\n   \n   and this is the second line\n'
            assert expect == actual
            actual = str(GoogleDocstring(f'{section}:\n    this is a single line\n', config))
            expect = f'.. {admonition}:: this is a single line\n'
            assert expect == actual

    def test_docstrings(self):
        if False:
            return 10
        config = Config(napoleon_use_param=False, napoleon_use_rtype=False, napoleon_use_keyword=False)
        for (docstring, expected) in self.docstrings:
            actual = str(GoogleDocstring(dedent(docstring), config))
            expected = dedent(expected)
            assert expected == actual

    def test_parameters_with_class_reference(self):
        if False:
            i = 10
            return i + 15
        docstring = 'Construct a new XBlock.\n\nThis class should only be used by runtimes.\n\nArguments:\n    runtime (:class:`~typing.Dict`\\[:class:`int`,:class:`str`\\]): Use it to\n        access the environment. It is available in XBlock code\n        as ``self.runtime``.\n\n    field_data (:class:`FieldData`): Interface used by the XBlock\n        fields to access their data from wherever it is persisted.\n\n    scope_ids (:class:`ScopeIds`): Identifiers needed to resolve scopes.\n\n'
        actual = str(GoogleDocstring(docstring))
        expected = 'Construct a new XBlock.\n\nThis class should only be used by runtimes.\n\n:param runtime: Use it to\n                access the environment. It is available in XBlock code\n                as ``self.runtime``.\n:type runtime: :class:`~typing.Dict`\\[:class:`int`,:class:`str`\\]\n:param field_data: Interface used by the XBlock\n                   fields to access their data from wherever it is persisted.\n:type field_data: :class:`FieldData`\n:param scope_ids: Identifiers needed to resolve scopes.\n:type scope_ids: :class:`ScopeIds`\n'
        assert expected == actual

    def test_attributes_with_class_reference(self):
        if False:
            for i in range(10):
                print('nop')
        docstring = 'Attributes:\n    in_attr(:class:`numpy.ndarray`): super-dooper attribute\n'
        actual = str(GoogleDocstring(docstring))
        expected = '.. attribute:: in_attr\n\n   super-dooper attribute\n\n   :type: :class:`numpy.ndarray`\n'
        assert expected == actual
        docstring = 'Attributes:\n    in_attr(numpy.ndarray): super-dooper attribute\n'
        actual = str(GoogleDocstring(docstring))
        expected = '.. attribute:: in_attr\n\n   super-dooper attribute\n\n   :type: numpy.ndarray\n'

    def test_attributes_with_use_ivar(self):
        if False:
            while True:
                i = 10
        docstring = 'Attributes:\n    foo (int): blah blah\n    bar (str): blah blah\n'
        config = Config(napoleon_use_ivar=True)
        actual = str(GoogleDocstring(docstring, config, obj=self.__class__))
        expected = ':ivar foo: blah blah\n:vartype foo: int\n:ivar bar: blah blah\n:vartype bar: str\n'
        assert expected == actual

    def test_code_block_in_returns_section(self):
        if False:
            print('Hello World!')
        docstring = '\nReturns:\n    foobar: foo::\n\n        codecode\n        codecode\n'
        expected = '\n:returns:\n\n          foo::\n\n              codecode\n              codecode\n:rtype: foobar\n'
        actual = str(GoogleDocstring(docstring))
        assert expected == actual

    def test_colon_in_return_type(self):
        if False:
            print('Hello World!')
        docstring = 'Example property.\n\nReturns:\n    :py:class:`~.module.submodule.SomeClass`: an example instance\n    if available, None if not available.\n'
        expected = 'Example property.\n\n:returns: an example instance\n          if available, None if not available.\n:rtype: :py:class:`~.module.submodule.SomeClass`\n'
        actual = str(GoogleDocstring(docstring))
        assert expected == actual

    def test_xrefs_in_return_type(self):
        if False:
            print('Hello World!')
        docstring = 'Example Function\n\nReturns:\n    :class:`numpy.ndarray`: A :math:`n \\times 2` array containing\n    a bunch of math items\n'
        expected = 'Example Function\n\n:returns: A :math:`n \\times 2` array containing\n          a bunch of math items\n:rtype: :class:`numpy.ndarray`\n'
        actual = str(GoogleDocstring(docstring))
        assert expected == actual

    def test_raises_types(self):
        if False:
            for i in range(10):
                print('nop')
        docstrings = [("\nExample Function\n\nRaises:\n    RuntimeError:\n        A setting wasn't specified, or was invalid.\n    ValueError:\n        Something something value error.\n    :py:class:`AttributeError`\n        errors for missing attributes.\n    ~InvalidDimensionsError\n        If the dimensions couldn't be parsed.\n    `InvalidArgumentsError`\n        If the arguments are invalid.\n    :exc:`~ValueError`\n        If the arguments are wrong.\n\n", "\nExample Function\n\n:raises RuntimeError: A setting wasn't specified, or was invalid.\n:raises ValueError: Something something value error.\n:raises AttributeError: errors for missing attributes.\n:raises ~InvalidDimensionsError: If the dimensions couldn't be parsed.\n:raises InvalidArgumentsError: If the arguments are invalid.\n:raises ~ValueError: If the arguments are wrong.\n"), ('\nExample Function\n\nRaises:\n    InvalidDimensionsError\n\n', '\nExample Function\n\n:raises InvalidDimensionsError:\n'), ('\nExample Function\n\nRaises:\n    Invalid Dimensions Error\n\n', '\nExample Function\n\n:raises Invalid Dimensions Error:\n'), ('\nExample Function\n\nRaises:\n    Invalid Dimensions Error: With description\n\n', '\nExample Function\n\n:raises Invalid Dimensions Error: With description\n'), ("\nExample Function\n\nRaises:\n    InvalidDimensionsError: If the dimensions couldn't be parsed.\n\n", "\nExample Function\n\n:raises InvalidDimensionsError: If the dimensions couldn't be parsed.\n"), ("\nExample Function\n\nRaises:\n    Invalid Dimensions Error: If the dimensions couldn't be parsed.\n\n", "\nExample Function\n\n:raises Invalid Dimensions Error: If the dimensions couldn't be parsed.\n"), ("\nExample Function\n\nRaises:\n    If the dimensions couldn't be parsed.\n\n", "\nExample Function\n\n:raises If the dimensions couldn't be parsed.:\n"), ('\nExample Function\n\nRaises:\n    :class:`exc.InvalidDimensionsError`\n\n', '\nExample Function\n\n:raises exc.InvalidDimensionsError:\n'), ("\nExample Function\n\nRaises:\n    :class:`exc.InvalidDimensionsError`: If the dimensions couldn't be parsed.\n\n", "\nExample Function\n\n:raises exc.InvalidDimensionsError: If the dimensions couldn't be parsed.\n"), ("\nExample Function\n\nRaises:\n    :class:`exc.InvalidDimensionsError`: If the dimensions couldn't be parsed,\n       then a :class:`exc.InvalidDimensionsError` will be raised.\n\n", "\nExample Function\n\n:raises exc.InvalidDimensionsError: If the dimensions couldn't be parsed,\n    then a :class:`exc.InvalidDimensionsError` will be raised.\n"), ("\nExample Function\n\nRaises:\n    :class:`exc.InvalidDimensionsError`: If the dimensions couldn't be parsed.\n    :class:`exc.InvalidArgumentsError`: If the arguments are invalid.\n\n", "\nExample Function\n\n:raises exc.InvalidDimensionsError: If the dimensions couldn't be parsed.\n:raises exc.InvalidArgumentsError: If the arguments are invalid.\n"), ('\nExample Function\n\nRaises:\n    :class:`exc.InvalidDimensionsError`\n    :class:`exc.InvalidArgumentsError`\n\n', '\nExample Function\n\n:raises exc.InvalidDimensionsError:\n:raises exc.InvalidArgumentsError:\n')]
        for (docstring, expected) in docstrings:
            actual = str(GoogleDocstring(docstring))
            assert expected == actual

    def test_kwargs_in_arguments(self):
        if False:
            for i in range(10):
                print('nop')
        docstring = 'Allows to create attributes binded to this device.\n\nSome other paragraph.\n\nCode sample for usage::\n\n  dev.bind(loopback=Loopback)\n  dev.loopback.configure()\n\nArguments:\n  **kwargs: name/class pairs that will create resource-managers\n    bound as instance attributes to this instance. See code\n    example above.\n'
        expected = 'Allows to create attributes binded to this device.\n\nSome other paragraph.\n\nCode sample for usage::\n\n  dev.bind(loopback=Loopback)\n  dev.loopback.configure()\n\n:param \\*\\*kwargs: name/class pairs that will create resource-managers\n                   bound as instance attributes to this instance. See code\n                   example above.\n'
        actual = str(GoogleDocstring(docstring))
        assert expected == actual

    def test_section_header_formatting(self):
        if False:
            for i in range(10):
                print('nop')
        docstrings = [('\nSummary line\n\nExample:\n    Multiline reStructuredText\n    literal code block\n\n', '\nSummary line\n\n.. rubric:: Example\n\nMultiline reStructuredText\nliteral code block\n'), ('\nSummary line\n\nExample::\n\n    Multiline reStructuredText\n    literal code block\n\n', '\nSummary line\n\nExample::\n\n    Multiline reStructuredText\n    literal code block\n'), ('\nSummary line\n\n:Example:\n\n    Multiline reStructuredText\n    literal code block\n\n', '\nSummary line\n\n:Example:\n\n    Multiline reStructuredText\n    literal code block\n')]
        for (docstring, expected) in docstrings:
            actual = str(GoogleDocstring(docstring))
            assert expected == actual

    def test_list_in_parameter_description(self):
        if False:
            for i in range(10):
                print('nop')
        docstring = 'One line summary.\n\nParameters:\n    no_list (int):\n    one_bullet_empty (int):\n        *\n    one_bullet_single_line (int):\n        - first line\n    one_bullet_two_lines (int):\n        +   first line\n            continued\n    two_bullets_single_line (int):\n        -  first line\n        -  second line\n    two_bullets_two_lines (int):\n        * first line\n          continued\n        * second line\n          continued\n    one_enumeration_single_line (int):\n        1.  first line\n    one_enumeration_two_lines (int):\n        1)   first line\n             continued\n    two_enumerations_one_line (int):\n        (iii) first line\n        (iv) second line\n    two_enumerations_two_lines (int):\n        a. first line\n           continued\n        b. second line\n           continued\n    one_definition_one_line (int):\n        item 1\n            first line\n    one_definition_two_lines (int):\n        item 1\n            first line\n            continued\n    two_definitions_one_line (int):\n        item 1\n            first line\n        item 2\n            second line\n    two_definitions_two_lines (int):\n        item 1\n            first line\n            continued\n        item 2\n            second line\n            continued\n    one_definition_blank_line (int):\n        item 1\n\n            first line\n\n            extra first line\n\n    two_definitions_blank_lines (int):\n        item 1\n\n            first line\n\n            extra first line\n\n        item 2\n\n            second line\n\n            extra second line\n\n    definition_after_inline_text (int): text line\n\n        item 1\n            first line\n\n    definition_after_normal_text (int):\n        text line\n\n        item 1\n            first line\n'
        expected = 'One line summary.\n\n:param no_list:\n:type no_list: int\n:param one_bullet_empty:\n                         *\n:type one_bullet_empty: int\n:param one_bullet_single_line:\n                               - first line\n:type one_bullet_single_line: int\n:param one_bullet_two_lines:\n                             +   first line\n                                 continued\n:type one_bullet_two_lines: int\n:param two_bullets_single_line:\n                                -  first line\n                                -  second line\n:type two_bullets_single_line: int\n:param two_bullets_two_lines:\n                              * first line\n                                continued\n                              * second line\n                                continued\n:type two_bullets_two_lines: int\n:param one_enumeration_single_line:\n                                    1.  first line\n:type one_enumeration_single_line: int\n:param one_enumeration_two_lines:\n                                  1)   first line\n                                       continued\n:type one_enumeration_two_lines: int\n:param two_enumerations_one_line:\n                                  (iii) first line\n                                  (iv) second line\n:type two_enumerations_one_line: int\n:param two_enumerations_two_lines:\n                                   a. first line\n                                      continued\n                                   b. second line\n                                      continued\n:type two_enumerations_two_lines: int\n:param one_definition_one_line:\n                                item 1\n                                    first line\n:type one_definition_one_line: int\n:param one_definition_two_lines:\n                                 item 1\n                                     first line\n                                     continued\n:type one_definition_two_lines: int\n:param two_definitions_one_line:\n                                 item 1\n                                     first line\n                                 item 2\n                                     second line\n:type two_definitions_one_line: int\n:param two_definitions_two_lines:\n                                  item 1\n                                      first line\n                                      continued\n                                  item 2\n                                      second line\n                                      continued\n:type two_definitions_two_lines: int\n:param one_definition_blank_line:\n                                  item 1\n\n                                      first line\n\n                                      extra first line\n:type one_definition_blank_line: int\n:param two_definitions_blank_lines:\n                                    item 1\n\n                                        first line\n\n                                        extra first line\n\n                                    item 2\n\n                                        second line\n\n                                        extra second line\n:type two_definitions_blank_lines: int\n:param definition_after_inline_text: text line\n\n                                     item 1\n                                         first line\n:type definition_after_inline_text: int\n:param definition_after_normal_text: text line\n\n                                     item 1\n                                         first line\n:type definition_after_normal_text: int\n'
        config = Config(napoleon_use_param=True)
        actual = str(GoogleDocstring(docstring, config))
        assert expected == actual
        expected = 'One line summary.\n\n:Parameters: * **no_list** (*int*)\n             * **one_bullet_empty** (*int*) --\n\n               *\n             * **one_bullet_single_line** (*int*) --\n\n               - first line\n             * **one_bullet_two_lines** (*int*) --\n\n               +   first line\n                   continued\n             * **two_bullets_single_line** (*int*) --\n\n               -  first line\n               -  second line\n             * **two_bullets_two_lines** (*int*) --\n\n               * first line\n                 continued\n               * second line\n                 continued\n             * **one_enumeration_single_line** (*int*) --\n\n               1.  first line\n             * **one_enumeration_two_lines** (*int*) --\n\n               1)   first line\n                    continued\n             * **two_enumerations_one_line** (*int*) --\n\n               (iii) first line\n               (iv) second line\n             * **two_enumerations_two_lines** (*int*) --\n\n               a. first line\n                  continued\n               b. second line\n                  continued\n             * **one_definition_one_line** (*int*) --\n\n               item 1\n                   first line\n             * **one_definition_two_lines** (*int*) --\n\n               item 1\n                   first line\n                   continued\n             * **two_definitions_one_line** (*int*) --\n\n               item 1\n                   first line\n               item 2\n                   second line\n             * **two_definitions_two_lines** (*int*) --\n\n               item 1\n                   first line\n                   continued\n               item 2\n                   second line\n                   continued\n             * **one_definition_blank_line** (*int*) --\n\n               item 1\n\n                   first line\n\n                   extra first line\n             * **two_definitions_blank_lines** (*int*) --\n\n               item 1\n\n                   first line\n\n                   extra first line\n\n               item 2\n\n                   second line\n\n                   extra second line\n             * **definition_after_inline_text** (*int*) -- text line\n\n               item 1\n                   first line\n             * **definition_after_normal_text** (*int*) -- text line\n\n               item 1\n                   first line\n'
        config = Config(napoleon_use_param=False)
        actual = str(GoogleDocstring(docstring, config))
        assert expected == actual

    def test_custom_generic_sections(self):
        if False:
            while True:
                i = 10
        docstrings = (('Really Important Details:\n    You should listen to me!\n', '.. rubric:: Really Important Details\n\nYou should listen to me!\n'), ('Sooper Warning:\n    Stop hitting yourself!\n', ':Warns: **Stop hitting yourself!**\n'), ('Params Style:\n    arg1 (int): Description of arg1\n    arg2 (str): Description of arg2\n\n', ':Params Style: * **arg1** (*int*) -- Description of arg1\n               * **arg2** (*str*) -- Description of arg2\n'), ('Returns Style:\n    description of custom section\n\n', ':Returns Style: description of custom section\n'))
        testConfig = Config(napoleon_custom_sections=['Really Important Details', ('Sooper Warning', 'warns'), ('Params Style', 'params_style'), ('Returns Style', 'returns_style')])
        for (docstring, expected) in docstrings:
            actual = str(GoogleDocstring(docstring, testConfig))
            assert expected == actual

    def test_noindex(self):
        if False:
            i = 10
            return i + 15
        docstring = '\nAttributes:\n    arg\n        description\n\nMethods:\n    func(i, j)\n        description\n'
        expected = '\n.. attribute:: arg\n   :no-index:\n\n   description\n\n.. method:: func(i, j)\n   :no-index:\n\n   \n   description\n'
        config = Config()
        actual = str(GoogleDocstring(docstring, config=config, app=None, what='module', options={'no-index': True}))
        assert expected == actual

    def test_keywords_with_types(self):
        if False:
            while True:
                i = 10
        docstring = 'Do as you please\n\nKeyword Args:\n    gotham_is_yours (None): shall interfere.\n'
        actual = str(GoogleDocstring(docstring))
        expected = 'Do as you please\n\n:keyword gotham_is_yours: shall interfere.\n:kwtype gotham_is_yours: None\n'
        assert expected == actual

    def test_pep526_annotations(self):
        if False:
            while True:
                i = 10
        config = Config(napoleon_attr_annotations=True)
        actual = str(GoogleDocstring(cleandoc(PEP526GoogleClass.__doc__), config, app=None, what='class', obj=PEP526GoogleClass))
        expected = 'Sample class with PEP 526 annotations and google docstring\n\n.. attribute:: attr1\n\n   Attr1 description.\n\n   :type: int\n\n.. attribute:: attr2\n\n   Attr2 description.\n\n   :type: str\n'
        assert expected == actual

    def test_preprocess_types(self):
        if False:
            while True:
                i = 10
        docstring = 'Do as you please\n\nYield:\n   str:Extended\n'
        actual = str(GoogleDocstring(docstring))
        expected = 'Do as you please\n\n:Yields: *str* -- Extended\n'
        assert expected == actual
        config = Config(napoleon_preprocess_types=True)
        actual = str(GoogleDocstring(docstring, config))
        expected = 'Do as you please\n\n:Yields: :py:class:`str` -- Extended\n'
        assert expected == actual

class TestNumpyDocstring:
    docstrings = [('Single line summary', 'Single line summary'), ('\n        Single line summary\n\n        Extended description\n\n        ', '\n        Single line summary\n\n        Extended description\n        '), ('\n        Single line summary\n\n        Parameters\n        ----------\n        arg1:str\n            Extended\n            description of arg1\n        ', '\n        Single line summary\n\n        :Parameters: **arg1** (:class:`str`) -- Extended\n                     description of arg1\n        '), ('\n        Single line summary\n\n        Parameters\n        ----------\n        arg1:str\n            Extended\n            description of arg1\n        arg2 : int\n            Extended\n            description of arg2\n\n        Keyword Arguments\n        -----------------\n          kwarg1:str\n              Extended\n              description of kwarg1\n          kwarg2 : int\n              Extended\n              description of kwarg2\n        ', '\n        Single line summary\n\n        :Parameters: * **arg1** (:class:`str`) -- Extended\n                       description of arg1\n                     * **arg2** (:class:`int`) -- Extended\n                       description of arg2\n\n        :Keyword Arguments: * **kwarg1** (:class:`str`) -- Extended\n                              description of kwarg1\n                            * **kwarg2** (:class:`int`) -- Extended\n                              description of kwarg2\n        '), ('\n        Single line summary\n\n        Return\n        ------\n        str\n            Extended\n            description of return value\n        ', '\n        Single line summary\n\n        :returns: :class:`str` -- Extended\n                  description of return value\n        '), ('\n        Single line summary\n\n        Returns\n        -------\n        str\n            Extended\n            description of return value\n        ', '\n        Single line summary\n\n        :returns: :class:`str` -- Extended\n                  description of return value\n        '), ('\n        Single line summary\n\n        Parameters\n        ----------\n        arg1:str\n             Extended description of arg1\n        *args:\n            Variable length argument list.\n        **kwargs:\n            Arbitrary keyword arguments.\n        ', '\n        Single line summary\n\n        :Parameters: * **arg1** (:class:`str`) -- Extended description of arg1\n                     * **\\*args** -- Variable length argument list.\n                     * **\\*\\*kwargs** -- Arbitrary keyword arguments.\n        '), ('\n        Single line summary\n\n        Parameters\n        ----------\n        arg1:str\n             Extended description of arg1\n        *args, **kwargs:\n            Variable length argument list and arbitrary keyword arguments.\n        ', '\n        Single line summary\n\n        :Parameters: * **arg1** (:class:`str`) -- Extended description of arg1\n                     * **\\*args, \\*\\*kwargs** -- Variable length argument list and arbitrary keyword arguments.\n        '), ('\n        Single line summary\n\n        Receive\n        -------\n        arg1:str\n            Extended\n            description of arg1\n        arg2 : int\n            Extended\n            description of arg2\n        ', '\n        Single line summary\n\n        :Receives: * **arg1** (:class:`str`) -- Extended\n                     description of arg1\n                   * **arg2** (:class:`int`) -- Extended\n                     description of arg2\n        '), ('\n        Single line summary\n\n        Receives\n        --------\n        arg1:str\n            Extended\n            description of arg1\n        arg2 : int\n            Extended\n            description of arg2\n        ', '\n        Single line summary\n\n        :Receives: * **arg1** (:class:`str`) -- Extended\n                     description of arg1\n                   * **arg2** (:class:`int`) -- Extended\n                     description of arg2\n        '), ('\n        Single line summary\n\n        Yield\n        -----\n        str\n            Extended\n            description of yielded value\n        ', '\n        Single line summary\n\n        :Yields: :class:`str` -- Extended\n                 description of yielded value\n        '), ('\n        Single line summary\n\n        Yields\n        ------\n        str\n            Extended\n            description of yielded value\n        ', '\n        Single line summary\n\n        :Yields: :class:`str` -- Extended\n                 description of yielded value\n        ')]

    def test_sphinx_admonitions(self):
        if False:
            while True:
                i = 10
        admonition_map = {'Attention': 'attention', 'Caution': 'caution', 'Danger': 'danger', 'Error': 'error', 'Hint': 'hint', 'Important': 'important', 'Note': 'note', 'Tip': 'tip', 'Todo': 'todo', 'Warning': 'warning', 'Warnings': 'warning'}
        config = Config()
        for (section, admonition) in admonition_map.items():
            actual = str(NumpyDocstring(f"{section}\n{'-' * len(section)}\n    this is the first line\n\n    and this is the second line\n", config))
            expect = f'.. {admonition}::\n\n   this is the first line\n   \n   and this is the second line\n'
            assert expect == actual
            actual = str(NumpyDocstring(f"{section}\n{'-' * len(section)}\n    this is a single line\n", config))
            expect = f'.. {admonition}:: this is a single line\n'
            assert expect == actual

    def test_docstrings(self):
        if False:
            while True:
                i = 10
        config = Config(napoleon_use_param=False, napoleon_use_rtype=False, napoleon_use_keyword=False, napoleon_preprocess_types=True)
        for (docstring, expected) in self.docstrings:
            actual = str(NumpyDocstring(dedent(docstring), config))
            expected = dedent(expected)
            assert expected == actual

    def test_type_preprocessor(self):
        if False:
            print('Hello World!')
        docstring = dedent('\n        Single line summary\n\n        Parameters\n        ----------\n        arg1:str\n            Extended\n            description of arg1\n        ')
        config = Config(napoleon_preprocess_types=False, napoleon_use_param=False)
        actual = str(NumpyDocstring(docstring, config))
        expected = dedent('\n        Single line summary\n\n        :Parameters: **arg1** (*str*) -- Extended\n                     description of arg1\n        ')
        assert expected == actual

    def test_parameters_with_class_reference(self):
        if False:
            while True:
                i = 10
        docstring = 'Parameters\n----------\nparam1 : :class:`MyClass <name.space.MyClass>` instance\n\nOther Parameters\n----------------\nparam2 : :class:`MyClass <name.space.MyClass>` instance\n\n'
        config = Config(napoleon_use_param=False)
        actual = str(NumpyDocstring(docstring, config))
        expected = ':Parameters: **param1** (:class:`MyClass <name.space.MyClass>` instance)\n\n:Other Parameters: **param2** (:class:`MyClass <name.space.MyClass>` instance)\n'
        assert expected == actual
        config = Config(napoleon_use_param=True)
        actual = str(NumpyDocstring(docstring, config))
        expected = ':param param1:\n:type param1: :class:`MyClass <name.space.MyClass>` instance\n\n:param param2:\n:type param2: :class:`MyClass <name.space.MyClass>` instance\n'
        assert expected == actual

    def test_multiple_parameters(self):
        if False:
            while True:
                i = 10
        docstring = 'Parameters\n----------\nx1, x2 : array_like\n    Input arrays, description of ``x1``, ``x2``.\n\n'
        config = Config(napoleon_use_param=False)
        actual = str(NumpyDocstring(docstring, config))
        expected = ':Parameters: **x1, x2** (*array_like*) -- Input arrays, description of ``x1``, ``x2``.\n'
        assert expected == actual
        config = Config(napoleon_use_param=True)
        actual = str(NumpyDocstring(dedent(docstring), config))
        expected = ':param x1: Input arrays, description of ``x1``, ``x2``.\n:type x1: array_like\n:param x2: Input arrays, description of ``x1``, ``x2``.\n:type x2: array_like\n'
        assert expected == actual

    def test_parameters_without_class_reference(self):
        if False:
            return 10
        docstring = 'Parameters\n----------\nparam1 : MyClass instance\n\n'
        config = Config(napoleon_use_param=False)
        actual = str(NumpyDocstring(docstring, config))
        expected = ':Parameters: **param1** (*MyClass instance*)\n'
        assert expected == actual
        config = Config(napoleon_use_param=True)
        actual = str(NumpyDocstring(dedent(docstring), config))
        expected = ':param param1:\n:type param1: MyClass instance\n'
        assert expected == actual

    def test_see_also_refs(self):
        if False:
            print('Hello World!')
        docstring = 'numpy.multivariate_normal(mean, cov, shape=None, spam=None)\n\nSee Also\n--------\nsome, other, funcs\notherfunc : relationship\n\n'
        actual = str(NumpyDocstring(docstring))
        expected = 'numpy.multivariate_normal(mean, cov, shape=None, spam=None)\n\n.. seealso::\n\n   :obj:`some`, :obj:`other`, :obj:`funcs`\n   \n   :obj:`otherfunc`\n       relationship\n'
        assert expected == actual
        docstring = 'numpy.multivariate_normal(mean, cov, shape=None, spam=None)\n\nSee Also\n--------\nsome, other, funcs\notherfunc : relationship\n\n'
        config = Config()
        app = mock.Mock()
        actual = str(NumpyDocstring(docstring, config, app, 'method'))
        expected = 'numpy.multivariate_normal(mean, cov, shape=None, spam=None)\n\n.. seealso::\n\n   :obj:`some`, :obj:`other`, :obj:`funcs`\n   \n   :obj:`otherfunc`\n       relationship\n'
        assert expected == actual
        docstring = 'numpy.multivariate_normal(mean, cov, shape=None, spam=None)\n\nSee Also\n--------\nsome, other, :func:`funcs`\notherfunc : relationship\n\n'
        translations = {'other': 'MyClass.other', 'otherfunc': ':func:`~my_package.otherfunc`'}
        config = Config(napoleon_type_aliases=translations)
        app = mock.Mock()
        actual = str(NumpyDocstring(docstring, config, app, 'method'))
        expected = 'numpy.multivariate_normal(mean, cov, shape=None, spam=None)\n\n.. seealso::\n\n   :obj:`some`, :obj:`MyClass.other`, :func:`funcs`\n   \n   :func:`~my_package.otherfunc`\n       relationship\n'
        assert expected == actual

    def test_colon_in_return_type(self):
        if False:
            return 10
        docstring = '\nSummary\n\nReturns\n-------\n:py:class:`~my_mod.my_class`\n    an instance of :py:class:`~my_mod.my_class`\n'
        expected = '\nSummary\n\n:returns: an instance of :py:class:`~my_mod.my_class`\n:rtype: :py:class:`~my_mod.my_class`\n'
        config = Config()
        app = mock.Mock()
        actual = str(NumpyDocstring(docstring, config, app, 'method'))
        assert expected == actual

    def test_underscore_in_attribute(self):
        if False:
            return 10
        docstring = '\nAttributes\n----------\n\narg_ : type\n    some description\n'
        expected = '\n:ivar arg_: some description\n:vartype arg_: type\n'
        config = Config(napoleon_use_ivar=True)
        app = mock.Mock()
        actual = str(NumpyDocstring(docstring, config, app, 'class'))
        assert expected == actual

    def test_underscore_in_attribute_strip_signature_backslash(self):
        if False:
            return 10
        docstring = '\nAttributes\n----------\n\narg_ : type\n    some description\n'
        expected = '\n:ivar arg\\_: some description\n:vartype arg\\_: type\n'
        config = Config(napoleon_use_ivar=True)
        config.strip_signature_backslash = True
        app = mock.Mock()
        actual = str(NumpyDocstring(docstring, config, app, 'class'))
        assert expected == actual

    def test_return_types(self):
        if False:
            return 10
        docstring = dedent('\n            Returns\n            -------\n            DataFrame\n                a dataframe\n        ')
        expected = dedent('\n           :returns: a dataframe\n           :rtype: :class:`~pandas.DataFrame`\n        ')
        translations = {'DataFrame': '~pandas.DataFrame'}
        config = Config(napoleon_use_param=True, napoleon_use_rtype=True, napoleon_preprocess_types=True, napoleon_type_aliases=translations)
        actual = str(NumpyDocstring(docstring, config))
        assert expected == actual

    def test_yield_types(self):
        if False:
            while True:
                i = 10
        docstring = dedent('\n            Example Function\n\n            Yields\n            ------\n            scalar or array-like\n                The result of the computation\n        ')
        expected = dedent('\n            Example Function\n\n            :Yields: :term:`scalar` or :class:`array-like <numpy.ndarray>` -- The result of the computation\n        ')
        translations = {'scalar': ':term:`scalar`', 'array-like': ':class:`array-like <numpy.ndarray>`'}
        config = Config(napoleon_type_aliases=translations, napoleon_preprocess_types=True)
        app = mock.Mock()
        actual = str(NumpyDocstring(docstring, config, app, 'method'))
        assert expected == actual

    def test_raises_types(self):
        if False:
            print('Hello World!')
        docstrings = [("\nExample Function\n\nRaises\n------\n  RuntimeError\n\n      A setting wasn't specified, or was invalid.\n  ValueError\n\n      Something something value error.\n\n", "\nExample Function\n\n:raises RuntimeError: A setting wasn't specified, or was invalid.\n:raises ValueError: Something something value error.\n"), ('\nExample Function\n\nRaises\n------\nInvalidDimensionsError\n\n', '\nExample Function\n\n:raises InvalidDimensionsError:\n'), ('\nExample Function\n\nRaises\n------\nInvalid Dimensions Error\n\n', '\nExample Function\n\n:raises Invalid Dimensions Error:\n'), ('\nExample Function\n\nRaises\n------\nInvalid Dimensions Error\n    With description\n\n', '\nExample Function\n\n:raises Invalid Dimensions Error: With description\n'), ("\nExample Function\n\nRaises\n------\nInvalidDimensionsError\n    If the dimensions couldn't be parsed.\n\n", "\nExample Function\n\n:raises InvalidDimensionsError: If the dimensions couldn't be parsed.\n"), ("\nExample Function\n\nRaises\n------\nInvalid Dimensions Error\n    If the dimensions couldn't be parsed.\n\n", "\nExample Function\n\n:raises Invalid Dimensions Error: If the dimensions couldn't be parsed.\n"), ("\nExample Function\n\nRaises\n------\nIf the dimensions couldn't be parsed.\n\n", "\nExample Function\n\n:raises If the dimensions couldn't be parsed.:\n"), ('\nExample Function\n\nRaises\n------\n:class:`exc.InvalidDimensionsError`\n\n', '\nExample Function\n\n:raises exc.InvalidDimensionsError:\n'), ("\nExample Function\n\nRaises\n------\n:class:`exc.InvalidDimensionsError`\n    If the dimensions couldn't be parsed.\n\n", "\nExample Function\n\n:raises exc.InvalidDimensionsError: If the dimensions couldn't be parsed.\n"), ("\nExample Function\n\nRaises\n------\n:class:`exc.InvalidDimensionsError`\n    If the dimensions couldn't be parsed,\n    then a :class:`exc.InvalidDimensionsError` will be raised.\n\n", "\nExample Function\n\n:raises exc.InvalidDimensionsError: If the dimensions couldn't be parsed,\n    then a :class:`exc.InvalidDimensionsError` will be raised.\n"), ("\nExample Function\n\nRaises\n------\n:class:`exc.InvalidDimensionsError`\n    If the dimensions couldn't be parsed.\n:class:`exc.InvalidArgumentsError`\n    If the arguments are invalid.\n\n", "\nExample Function\n\n:raises exc.InvalidDimensionsError: If the dimensions couldn't be parsed.\n:raises exc.InvalidArgumentsError: If the arguments are invalid.\n"), ("\nExample Function\n\nRaises\n------\nCustomError\n    If the dimensions couldn't be parsed.\n\n", "\nExample Function\n\n:raises package.CustomError: If the dimensions couldn't be parsed.\n"), ("\nExample Function\n\nRaises\n------\nAnotherError\n    If the dimensions couldn't be parsed.\n\n", "\nExample Function\n\n:raises ~package.AnotherError: If the dimensions couldn't be parsed.\n"), ('\nExample Function\n\nRaises\n------\n:class:`exc.InvalidDimensionsError`\n:class:`exc.InvalidArgumentsError`\n\n', '\nExample Function\n\n:raises exc.InvalidDimensionsError:\n:raises exc.InvalidArgumentsError:\n')]
        for (docstring, expected) in docstrings:
            translations = {'CustomError': 'package.CustomError', 'AnotherError': ':py:exc:`~package.AnotherError`'}
            config = Config(napoleon_type_aliases=translations, napoleon_preprocess_types=True)
            app = mock.Mock()
            actual = str(NumpyDocstring(docstring, config, app, 'method'))
            assert expected == actual

    def test_xrefs_in_return_type(self):
        if False:
            for i in range(10):
                print('nop')
        docstring = '\nExample Function\n\nReturns\n-------\n:class:`numpy.ndarray`\n    A :math:`n \\times 2` array containing\n    a bunch of math items\n'
        expected = '\nExample Function\n\n:returns: A :math:`n \\times 2` array containing\n          a bunch of math items\n:rtype: :class:`numpy.ndarray`\n'
        config = Config()
        app = mock.Mock()
        actual = str(NumpyDocstring(docstring, config, app, 'method'))
        assert expected == actual

    def test_section_header_underline_length(self):
        if False:
            return 10
        docstrings = [('\nSummary line\n\nExample\n-\nMultiline example\nbody\n\n', '\nSummary line\n\nExample\n-\nMultiline example\nbody\n'), ('\nSummary line\n\nExample\n--\nMultiline example\nbody\n\n', '\nSummary line\n\n.. rubric:: Example\n\nMultiline example\nbody\n'), ('\nSummary line\n\nExample\n-------\nMultiline example\nbody\n\n', '\nSummary line\n\n.. rubric:: Example\n\nMultiline example\nbody\n'), ('\nSummary line\n\nExample\n------------\nMultiline example\nbody\n\n', '\nSummary line\n\n.. rubric:: Example\n\nMultiline example\nbody\n')]
        for (docstring, expected) in docstrings:
            actual = str(NumpyDocstring(docstring))
            assert expected == actual

    def test_list_in_parameter_description(self):
        if False:
            while True:
                i = 10
        docstring = 'One line summary.\n\nParameters\n----------\nno_list : int\none_bullet_empty : int\n    *\none_bullet_single_line : int\n    - first line\none_bullet_two_lines : int\n    +   first line\n        continued\ntwo_bullets_single_line : int\n    -  first line\n    -  second line\ntwo_bullets_two_lines : int\n    * first line\n      continued\n    * second line\n      continued\none_enumeration_single_line : int\n    1.  first line\none_enumeration_two_lines : int\n    1)   first line\n         continued\ntwo_enumerations_one_line : int\n    (iii) first line\n    (iv) second line\ntwo_enumerations_two_lines : int\n    a. first line\n       continued\n    b. second line\n       continued\none_definition_one_line : int\n    item 1\n        first line\none_definition_two_lines : int\n    item 1\n        first line\n        continued\ntwo_definitions_one_line : int\n    item 1\n        first line\n    item 2\n        second line\ntwo_definitions_two_lines : int\n    item 1\n        first line\n        continued\n    item 2\n        second line\n        continued\none_definition_blank_line : int\n    item 1\n\n        first line\n\n        extra first line\n\ntwo_definitions_blank_lines : int\n    item 1\n\n        first line\n\n        extra first line\n\n    item 2\n\n        second line\n\n        extra second line\n\ndefinition_after_normal_text : int\n    text line\n\n    item 1\n        first line\n'
        expected = 'One line summary.\n\n:param no_list:\n:type no_list: int\n:param one_bullet_empty:\n                         *\n:type one_bullet_empty: int\n:param one_bullet_single_line:\n                               - first line\n:type one_bullet_single_line: int\n:param one_bullet_two_lines:\n                             +   first line\n                                 continued\n:type one_bullet_two_lines: int\n:param two_bullets_single_line:\n                                -  first line\n                                -  second line\n:type two_bullets_single_line: int\n:param two_bullets_two_lines:\n                              * first line\n                                continued\n                              * second line\n                                continued\n:type two_bullets_two_lines: int\n:param one_enumeration_single_line:\n                                    1.  first line\n:type one_enumeration_single_line: int\n:param one_enumeration_two_lines:\n                                  1)   first line\n                                       continued\n:type one_enumeration_two_lines: int\n:param two_enumerations_one_line:\n                                  (iii) first line\n                                  (iv) second line\n:type two_enumerations_one_line: int\n:param two_enumerations_two_lines:\n                                   a. first line\n                                      continued\n                                   b. second line\n                                      continued\n:type two_enumerations_two_lines: int\n:param one_definition_one_line:\n                                item 1\n                                    first line\n:type one_definition_one_line: int\n:param one_definition_two_lines:\n                                 item 1\n                                     first line\n                                     continued\n:type one_definition_two_lines: int\n:param two_definitions_one_line:\n                                 item 1\n                                     first line\n                                 item 2\n                                     second line\n:type two_definitions_one_line: int\n:param two_definitions_two_lines:\n                                  item 1\n                                      first line\n                                      continued\n                                  item 2\n                                      second line\n                                      continued\n:type two_definitions_two_lines: int\n:param one_definition_blank_line:\n                                  item 1\n\n                                      first line\n\n                                      extra first line\n:type one_definition_blank_line: int\n:param two_definitions_blank_lines:\n                                    item 1\n\n                                        first line\n\n                                        extra first line\n\n                                    item 2\n\n                                        second line\n\n                                        extra second line\n:type two_definitions_blank_lines: int\n:param definition_after_normal_text: text line\n\n                                     item 1\n                                         first line\n:type definition_after_normal_text: int\n'
        config = Config(napoleon_use_param=True)
        actual = str(NumpyDocstring(docstring, config))
        assert expected == actual
        expected = 'One line summary.\n\n:Parameters: * **no_list** (:class:`int`)\n             * **one_bullet_empty** (:class:`int`) --\n\n               *\n             * **one_bullet_single_line** (:class:`int`) --\n\n               - first line\n             * **one_bullet_two_lines** (:class:`int`) --\n\n               +   first line\n                   continued\n             * **two_bullets_single_line** (:class:`int`) --\n\n               -  first line\n               -  second line\n             * **two_bullets_two_lines** (:class:`int`) --\n\n               * first line\n                 continued\n               * second line\n                 continued\n             * **one_enumeration_single_line** (:class:`int`) --\n\n               1.  first line\n             * **one_enumeration_two_lines** (:class:`int`) --\n\n               1)   first line\n                    continued\n             * **two_enumerations_one_line** (:class:`int`) --\n\n               (iii) first line\n               (iv) second line\n             * **two_enumerations_two_lines** (:class:`int`) --\n\n               a. first line\n                  continued\n               b. second line\n                  continued\n             * **one_definition_one_line** (:class:`int`) --\n\n               item 1\n                   first line\n             * **one_definition_two_lines** (:class:`int`) --\n\n               item 1\n                   first line\n                   continued\n             * **two_definitions_one_line** (:class:`int`) --\n\n               item 1\n                   first line\n               item 2\n                   second line\n             * **two_definitions_two_lines** (:class:`int`) --\n\n               item 1\n                   first line\n                   continued\n               item 2\n                   second line\n                   continued\n             * **one_definition_blank_line** (:class:`int`) --\n\n               item 1\n\n                   first line\n\n                   extra first line\n             * **two_definitions_blank_lines** (:class:`int`) --\n\n               item 1\n\n                   first line\n\n                   extra first line\n\n               item 2\n\n                   second line\n\n                   extra second line\n             * **definition_after_normal_text** (:class:`int`) -- text line\n\n               item 1\n                   first line\n'
        config = Config(napoleon_use_param=False, napoleon_preprocess_types=True)
        actual = str(NumpyDocstring(docstring, config))
        assert expected == actual

    def test_token_type(self):
        if False:
            return 10
        tokens = (('1', 'literal'), ('-4.6', 'literal'), ('2j', 'literal'), ("'string'", 'literal'), ('"another_string"', 'literal'), ('{1, 2}', 'literal'), ("{'va{ue', 'set'}", 'literal'), ('optional', 'control'), ('default', 'control'), (', ', 'delimiter'), (' of ', 'delimiter'), (' or ', 'delimiter'), (': ', 'delimiter'), ('True', 'obj'), ('None', 'obj'), ('name', 'obj'), (':py:class:`Enum`', 'reference'))
        for (token, expected) in tokens:
            actual = _token_type(token)
            assert expected == actual

    def test_tokenize_type_spec(self):
        if False:
            i = 10
            return i + 15
        specs = ('str', 'defaultdict', 'int, float, or complex', 'int or float or None, optional', 'list of list of int or float, optional', 'tuple of list of str, float, or int', '{"F", "C", "N"}', "{'F', 'C', 'N'}, default: 'F'", "{'F', 'C', 'N or C'}, default 'F'", "str, default: 'F or C'", 'int, default: None', 'int, default None', 'int, default :obj:`None`', '"ma{icious"', "'with \\'quotes\\''")
        tokens = (['str'], ['defaultdict'], ['int', ', ', 'float', ', or ', 'complex'], ['int', ' or ', 'float', ' or ', 'None', ', ', 'optional'], ['list', ' of ', 'list', ' of ', 'int', ' or ', 'float', ', ', 'optional'], ['tuple', ' of ', 'list', ' of ', 'str', ', ', 'float', ', or ', 'int'], ['{', '"F"', ', ', '"C"', ', ', '"N"', '}'], ['{', "'F'", ', ', "'C'", ', ', "'N'", '}', ', ', 'default', ': ', "'F'"], ['{', "'F'", ', ', "'C'", ', ', "'N or C'", '}', ', ', 'default', ' ', "'F'"], ['str', ', ', 'default', ': ', "'F or C'"], ['int', ', ', 'default', ': ', 'None'], ['int', ', ', 'default', ' ', 'None'], ['int', ', ', 'default', ' ', ':obj:`None`'], ['"ma{icious"'], ["'with \\'quotes\\''"])
        for (spec, expected) in zip(specs, tokens):
            actual = _tokenize_type_spec(spec)
            assert expected == actual

    def test_recombine_set_tokens(self):
        if False:
            print('Hello World!')
        tokens = (['{', '1', ', ', '2', '}'], ['{', '"F"', ', ', '"C"', ', ', '"N"', '}', ', ', 'optional'], ['{', "'F'", ', ', "'C'", ', ', "'N'", '}', ', ', 'default', ': ', 'None'], ['{', "'F'", ', ', "'C'", ', ', "'N'", '}', ', ', 'default', ' ', 'None'])
        combined_tokens = (['{1, 2}'], ['{"F", "C", "N"}', ', ', 'optional'], ["{'F', 'C', 'N'}", ', ', 'default', ': ', 'None'], ["{'F', 'C', 'N'}", ', ', 'default', ' ', 'None'])
        for (tokens_, expected) in zip(tokens, combined_tokens):
            actual = _recombine_set_tokens(tokens_)
            assert expected == actual

    def test_recombine_set_tokens_invalid(self):
        if False:
            print('Hello World!')
        tokens = (['{', '1', ', ', '2'], ['"F"', ', ', '"C"', ', ', '"N"', '}', ', ', 'optional'], ['{', '1', ', ', '2', ', ', 'default', ': ', 'None'])
        combined_tokens = (['{1, 2'], ['"F"', ', ', '"C"', ', ', '"N"', '}', ', ', 'optional'], ['{1, 2', ', ', 'default', ': ', 'None'])
        for (tokens_, expected) in zip(tokens, combined_tokens):
            actual = _recombine_set_tokens(tokens_)
            assert expected == actual

    def test_convert_numpy_type_spec(self):
        if False:
            print('Hello World!')
        translations = {'DataFrame': 'pandas.DataFrame'}
        specs = ('', 'optional', 'str, optional', 'int or float or None, default: None', 'list of tuple of str, optional', 'int, default None', '{"F", "C", "N"}', "{'F', 'C', 'N'}, default: 'N'", "{'F', 'C', 'N'}, default 'N'", 'DataFrame, optional')
        converted = ('', '*optional*', ':class:`str`, *optional*', ':class:`int` or :class:`float` or :obj:`None`, *default*: :obj:`None`', ':class:`list` of :class:`tuple` of :class:`str`, *optional*', ':class:`int`, *default* :obj:`None`', '``{"F", "C", "N"}``', "``{'F', 'C', 'N'}``, *default*: ``'N'``", "``{'F', 'C', 'N'}``, *default* ``'N'``", ':class:`pandas.DataFrame`, *optional*')
        for (spec, expected) in zip(specs, converted):
            actual = _convert_numpy_type_spec(spec, translations=translations)
            assert expected == actual

    def test_parameter_types(self):
        if False:
            i = 10
            return i + 15
        docstring = dedent('            Parameters\n            ----------\n            param1 : DataFrame\n                the data to work on\n            param2 : int or float or None, optional\n                a parameter with different types\n            param3 : dict-like, optional\n                a optional mapping\n            param4 : int or float or None, optional\n                a optional parameter with different types\n            param5 : {"F", "C", "N"}, optional\n                a optional parameter with fixed values\n            param6 : int, default None\n                different default format\n            param7 : mapping of hashable to str, optional\n                a optional mapping\n            param8 : ... or Ellipsis\n                ellipsis\n            param9 : tuple of list of int\n                a parameter with tuple of list of int\n        ')
        expected = dedent('            :param param1: the data to work on\n            :type param1: :class:`DataFrame`\n            :param param2: a parameter with different types\n            :type param2: :class:`int` or :class:`float` or :obj:`None`, *optional*\n            :param param3: a optional mapping\n            :type param3: :term:`dict-like <mapping>`, *optional*\n            :param param4: a optional parameter with different types\n            :type param4: :class:`int` or :class:`float` or :obj:`None`, *optional*\n            :param param5: a optional parameter with fixed values\n            :type param5: ``{"F", "C", "N"}``, *optional*\n            :param param6: different default format\n            :type param6: :class:`int`, *default* :obj:`None`\n            :param param7: a optional mapping\n            :type param7: :term:`mapping` of :term:`hashable` to :class:`str`, *optional*\n            :param param8: ellipsis\n            :type param8: :obj:`... <Ellipsis>` or :obj:`Ellipsis`\n            :param param9: a parameter with tuple of list of int\n            :type param9: :class:`tuple` of :class:`list` of :class:`int`\n        ')
        translations = {'dict-like': ':term:`dict-like <mapping>`', 'mapping': ':term:`mapping`', 'hashable': ':term:`hashable`'}
        config = Config(napoleon_use_param=True, napoleon_use_rtype=True, napoleon_preprocess_types=True, napoleon_type_aliases=translations)
        actual = str(NumpyDocstring(docstring, config))
        assert expected == actual

    def test_token_type_invalid(self, warning):
        if False:
            for i in range(10):
                print('nop')
        tokens = ('{1, 2', '}', "'abc", "def'", '"ghi', 'jkl"')
        errors = ('.+: invalid value set \\(missing closing brace\\):', '.+: invalid value set \\(missing opening brace\\):', '.+: malformed string literal \\(missing closing quote\\):', '.+: malformed string literal \\(missing opening quote\\):', '.+: malformed string literal \\(missing closing quote\\):', '.+: malformed string literal \\(missing opening quote\\):')
        for (token, error) in zip(tokens, errors):
            try:
                _token_type(token)
            finally:
                raw_warnings = warning.getvalue()
                warnings = [w for w in raw_warnings.split('\n') if w.strip()]
                assert len(warnings) == 1
                assert re.compile(error).match(warnings[0])
                warning.truncate(0)

    @pytest.mark.parametrize(('name', 'expected'), [('x, y, z', 'x, y, z'), ('*args, **kwargs', '\\*args, \\*\\*kwargs'), ('*x, **y', '\\*x, \\*\\*y')])
    def test_escape_args_and_kwargs(self, name, expected):
        if False:
            i = 10
            return i + 15
        numpy_docstring = NumpyDocstring('')
        actual = numpy_docstring._escape_args_and_kwargs(name)
        assert actual == expected

    def test_pep526_annotations(self):
        if False:
            for i in range(10):
                print('nop')
        config = Config(napoleon_attr_annotations=True)
        actual = str(NumpyDocstring(cleandoc(PEP526NumpyClass.__doc__), config, app=None, what='class', obj=PEP526NumpyClass))
        expected = 'Sample class with PEP 526 annotations and numpy docstring\n\n.. attribute:: attr1\n\n   Attr1 description\n\n   :type: int\n\n.. attribute:: attr2\n\n   Attr2 description\n\n   :type: str\n'
        print(actual)
        assert expected == actual

@pytest.mark.sphinx('text', testroot='ext-napoleon', confoverrides={'autodoc_typehints': 'description', 'autodoc_typehints_description_target': 'all'})
def test_napoleon_and_autodoc_typehints_description_all(app, status, warning):
    if False:
        print('Hello World!')
    app.build()
    content = (app.outdir / 'typehints.txt').read_text(encoding='utf-8')
    assert content == 'typehints\n*********\n\nmypackage.typehints.hello(x, *args, **kwargs)\n\n   Parameters:\n      * **x** (*int*) -- X\n\n      * ***args** (*int*) -- Additional arguments.\n\n      * ****kwargs** (*int*) -- Extra arguments.\n\n   Return type:\n      None\n'

@pytest.mark.sphinx('text', testroot='ext-napoleon', confoverrides={'autodoc_typehints': 'description', 'autodoc_typehints_description_target': 'documented_params'})
def test_napoleon_and_autodoc_typehints_description_documented_params(app, status, warning):
    if False:
        print('Hello World!')
    app.build()
    content = (app.outdir / 'typehints.txt').read_text(encoding='utf-8')
    assert content == 'typehints\n*********\n\nmypackage.typehints.hello(x, *args, **kwargs)\n\n   Parameters:\n      * **x** (*int*) -- X\n\n      * ***args** (*int*) -- Additional arguments.\n\n      * ****kwargs** (*int*) -- Extra arguments.\n'