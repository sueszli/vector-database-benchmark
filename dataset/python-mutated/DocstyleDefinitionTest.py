import unittest
from unittest.mock import patch
from coalib.bearlib.languages.documentation.DocstyleDefinition import DocstyleDefinition

class DocstyleDefinitionTest(unittest.TestCase):
    Metadata = DocstyleDefinition.Metadata
    ClassPadding = DocstyleDefinition.ClassPadding
    FunctionPadding = DocstyleDefinition.FunctionPadding
    DocstringTypeRegex = DocstyleDefinition.DocstringTypeRegex
    dummy_metadata = Metadata(':param ', ':', ':raises ', ':', ':return:')
    dummy_class_padding = ClassPadding(1, 1)
    dummy_function_padding = FunctionPadding(0, 1)
    dummy_docstring_type_regex = DocstringTypeRegex('class', 'def')
    dummy_docstring_position = 'top'

    def test_fail_instantiation(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            DocstyleDefinition('PYTHON', 'doxyGEN', (('##', '#'),), self.dummy_metadata, self.dummy_class_padding, self.dummy_function_padding, self.dummy_docstring_type_regex, self.dummy_docstring_position)
        with self.assertRaises(ValueError):
            DocstyleDefinition('WEIRD-PY', 'schloxygen', (('##+', 'x', 'y', 'z'),), self.dummy_metadata, self.dummy_class_padding, self.dummy_function_padding, self.dummy_docstring_type_regex, self.dummy_docstring_position)
        with self.assertRaises(ValueError):
            DocstyleDefinition('PYTHON', 'doxygen', (('##', '', '#'), ('"""', '"""')), self.dummy_metadata, self.dummy_class_padding, self.dummy_function_padding, self.dummy_docstring_type_regex, self.dummy_docstring_position)
        with self.assertRaises(TypeError):
            DocstyleDefinition(123, ['doxygen'], ('"""', '"""'), self.dummy_metadata, self.dummy_class_padding, self.dummy_function_padding, self.dummy_docstring_type_regex, self.dummy_docstring_position)
        with self.assertRaises(TypeError):
            DocstyleDefinition('language', ['doxygen'], ('"""', '"""'), 'metdata', 'clpading', 'dfpadding', 'kind')

    def test_properties(self):
        if False:
            while True:
                i = 10
        uut = DocstyleDefinition('C', 'doxygen', (('/**', '*', '*/'),), self.dummy_metadata, self.dummy_class_padding, self.dummy_function_padding, self.dummy_docstring_type_regex, self.dummy_docstring_position)
        self.assertEqual(uut.language, 'c')
        self.assertEqual(uut.docstyle, 'doxygen')
        self.assertEqual(uut.markers, (('/**', '*', '*/'),))
        self.assertEqual(uut.metadata, self.dummy_metadata)
        self.assertEqual(uut.class_padding, self.dummy_class_padding)
        self.assertEqual(uut.function_padding, self.dummy_function_padding)
        self.assertEqual(uut.docstring_type_regex, self.dummy_docstring_type_regex)
        self.assertEqual(uut.docstring_position, self.dummy_docstring_position)
        uut = DocstyleDefinition('PYTHON', 'doxyGEN', [('##', '', '#')], self.dummy_metadata, self.dummy_class_padding, self.dummy_function_padding, self.dummy_docstring_type_regex, self.dummy_docstring_position)
        self.assertEqual(uut.language, 'python')
        self.assertEqual(uut.docstyle, 'doxygen')
        self.assertEqual(uut.markers, (('##', '', '#'),))
        self.assertEqual(uut.metadata, self.dummy_metadata)
        uut = DocstyleDefinition('I2C', 'my-custom-tool', (['~~', '/~', '/~'], ('>!', '>>', '>>')), self.dummy_metadata, self.dummy_class_padding, self.dummy_function_padding, self.dummy_docstring_type_regex, self.dummy_docstring_position)
        self.assertEqual(uut.language, 'i2c')
        self.assertEqual(uut.docstyle, 'my-custom-tool')
        self.assertEqual(uut.markers, (('~~', '/~', '/~'), ('>!', '>>', '>>')))
        self.assertEqual(uut.metadata, self.dummy_metadata)
        uut = DocstyleDefinition('Cpp', 'doxygen', ('~~', '/~', '/~'), self.dummy_metadata, self.dummy_class_padding, self.dummy_function_padding, self.dummy_docstring_type_regex, self.dummy_docstring_position)
        self.assertEqual(uut.language, 'cpp')
        self.assertEqual(uut.docstyle, 'doxygen')
        self.assertEqual(uut.markers, (('~~', '/~', '/~'),))
        self.assertEqual(uut.metadata, self.dummy_metadata)

    def test_load(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(FileNotFoundError):
            next(DocstyleDefinition.load('PYTHON', 'INVALID'))
        with self.assertRaises(KeyError):
            next(DocstyleDefinition.load('bake-a-cake', 'default'))
        with self.assertRaises(TypeError):
            next(DocstyleDefinition.load(123, ['list']))
        result = DocstyleDefinition.load('PYTHON3', 'default')
        self.assertEqual(result.language, 'python3')
        self.assertEqual(result.docstyle, 'default')
        self.assertEqual(result.markers, (('"""', '', '"""'),))
        self.assertEqual(result.metadata, self.dummy_metadata)

    def test_get_available_definitions(self):
        if False:
            i = 10
            return i + 15
        expected = {('default', 'python'), ('default', 'python3'), ('default', 'java'), ('doxygen', 'c'), ('doxygen', 'cpp'), ('doxygen', 'cs'), ('doxygen', 'fortran'), ('doxygen', 'java'), ('doxygen', 'python'), ('doxygen', 'python3'), ('doxygen', 'tcl'), ('doxygen', 'vhdl'), ('doxygen', 'php'), ('doxygen', 'objective-c')}
        real = set(DocstyleDefinition.get_available_definitions())
        self.assertTrue(expected.issubset(real))

    @patch('coalib.bearlib.languages.documentation.DocstyleDefinition.iglob')
    @patch('coalib.bearlib.languages.documentation.DocstyleDefinition.ConfParser')
    def test_get_available_definitions_on_wrong_files(self, confparser_mock, iglob_mock):
        if False:
            print('Hello World!')
        confparser_instance_mock = confparser_mock.return_value
        confparser_instance_mock.parse.return_value = ['X']
        iglob_mock.return_value = ['some/CUSTOMSTYLE.coalang', 'SOME/xlang.coalang']
        self.assertEqual(list(DocstyleDefinition.get_available_definitions()), [('xlang', 'x')])