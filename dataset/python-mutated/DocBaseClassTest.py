import unittest
from coalib.bearlib.languages.documentation.DocstyleDefinition import DocstyleDefinition
from coalib.bearlib.languages.documentation.DocumentationComment import DocumentationComment, MalformedComment
from coalib.bearlib.languages.documentation.DocBaseClass import DocBaseClass
from tests.bearlib.languages.documentation.TestUtils import load_testdata
from coalib.results.TextPosition import TextPosition
from coalib.results.TextRange import TextRange
from coalib.results.Diff import Diff
from textwrap import dedent

class DocBaseClassTest(unittest.TestCase):

    def test_DocBaseClass_extraction_invalid_input(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(FileNotFoundError):
            tuple(DocBaseClass.extract('', 'PYTHON', 'INVALID'))

    def test_DocBaseClass_extraction_C(self):
        if False:
            return 10
        data = load_testdata('data.c')
        with self.assertRaises(KeyError):
            tuple(DocBaseClass.extract(data, 'C', 'default'))
        docstyle_C_doxygen = DocstyleDefinition.load('C', 'doxygen')
        expected_results = (DocumentationComment('\n This is the main function.\n\n @returns Your favorite number.\n', docstyle_C_doxygen, '', docstyle_C_doxygen.markers[0], TextPosition(3, 1)), DocumentationComment('\n Preserves alignment\n - Main item\n   - sub item\n     - sub sub item\n', docstyle_C_doxygen, '', docstyle_C_doxygen.markers[2], TextPosition(15, 1)), DocumentationComment(' ABC\n    Another type of comment\n\n    ...', docstyle_C_doxygen, '', docstyle_C_doxygen.markers[1], TextPosition(23, 1)), DocumentationComment(' foobar = barfoo.\n @param x whatever...\n', docstyle_C_doxygen, '', docstyle_C_doxygen.markers[0], TextPosition(28, 1)))
        self.assertEqual(tuple(DocBaseClass.extract(data, 'C', 'doxygen')), expected_results)

    def test_DocBaseClass_extraction_C_2(self):
        if False:
            while True:
                i = 10
        data = ['/** my main description\n', ' * continues here */']
        docstyle_C_doxygen = DocstyleDefinition.load('C', 'doxygen')
        self.assertEqual(list(DocBaseClass.extract(data, 'C', 'doxygen')), [DocumentationComment(' my main description\n continues here', docstyle_C_doxygen, '', docstyle_C_doxygen.markers[0], TextPosition(1, 1))])

    def test_DocBaseClass_extraction_CPP(self):
        if False:
            for i in range(10):
                print('nop')
        data = load_testdata('data.cpp')
        with self.assertRaises(KeyError):
            tuple(DocBaseClass.extract(data, 'CPP', 'default'))
        docstyle_CPP_doxygen = DocstyleDefinition.load('CPP', 'doxygen')
        self.assertEqual(tuple(DocBaseClass.extract(data, 'CPP', 'doxygen')), (DocumentationComment('\n This is the main function.\n @returns Exit code.\n          Or any other number.\n', docstyle_CPP_doxygen, '', docstyle_CPP_doxygen.markers[0], TextPosition(4, 1)), DocumentationComment(' foobar\n @param xyz\n', docstyle_CPP_doxygen, '', docstyle_CPP_doxygen.markers[0], TextPosition(15, 1)), DocumentationComment(' Some alternate style of documentation\n', docstyle_CPP_doxygen, '', docstyle_CPP_doxygen.markers[4], TextPosition(22, 1)), DocumentationComment(' ends instantly', docstyle_CPP_doxygen, '\t', docstyle_CPP_doxygen.markers[0], TextPosition(26, 2)), DocumentationComment(' Should work\n\n even without a function standing below.\n\n @param foo WHAT PARAM PLEASE!?\n', docstyle_CPP_doxygen, '', docstyle_CPP_doxygen.markers[4], TextPosition(32, 1))))

    def test_DocBaseClass_CPP_2(self):
        if False:
            print('Hello World!')
        data = load_testdata('data2.cpp')
        docstyle_CPP_doxygen = DocstyleDefinition.load('CPP', 'doxygen')
        self.assertEqual(tuple(DocBaseClass.extract(data, 'CPP', 'doxygen')), (DocumentationComment('module comment\n hello world\n', docstyle_CPP_doxygen, '', docstyle_CPP_doxygen.markers[0], TextPosition(1, 1)),))

    def test_DocBaseClass_PYTHON3(self):
        if False:
            print('Hello World!')
        data = load_testdata('data.py')
        docstyle_PYTHON3_default = DocstyleDefinition.load('PYTHON3', 'default')
        docstyle_PYTHON3_doxygen = DocstyleDefinition.load('PYTHON3', 'doxygen')
        expected = (DocumentationComment('\nModule description.\n\nSome more foobar-like text.\n', docstyle_PYTHON3_default, '', docstyle_PYTHON3_default.markers[0], TextPosition(1, 1)), DocumentationComment('\nA nice and neat way of documenting code.\n:param radius: The explosion radius.\n', docstyle_PYTHON3_default, ' ' * 4, docstyle_PYTHON3_default.markers[0], TextPosition(8, 5)), DocumentationComment('\nA function that returns 55.\n', docstyle_PYTHON3_default, ' ' * 8, docstyle_PYTHON3_default.markers[0], TextPosition(13, 9)), DocumentationComment('\nDocstring with layouted text.\n\n    layouts inside docs are preserved for these documentation styles.\nthis is intended.\n', docstyle_PYTHON3_default, '', docstyle_PYTHON3_default.markers[0], TextPosition(19, 1)), DocumentationComment(' Docstring directly besides triple quotes.\n    Continues here. ', docstyle_PYTHON3_default, '', docstyle_PYTHON3_default.markers[0], TextPosition(26, 1)), DocumentationComment('super\n nicely\nshort', docstyle_PYTHON3_default, '', docstyle_PYTHON3_default.markers[0], TextPosition(40, 1)), DocumentationComment('\nA bad indented docstring\n    Improper indentation.\n:param impact: The force of Impact.\n', docstyle_PYTHON3_default, ' ' * 4, docstyle_PYTHON3_default.markers[0], TextPosition(45, 5)))
        self.assertEqual(tuple(DocBaseClass.extract(data, 'PYTHON3', 'default')), expected)
        expected = list((DocumentationComment(r.documentation, docstyle_PYTHON3_doxygen, r.indent, r.marker, r.position) for r in expected))
        expected.insert(5, DocumentationComment(' Alternate documentation style in doxygen.\n  Subtext\n More subtext (not correctly aligned)\n      sub-sub-text\n\n', docstyle_PYTHON3_doxygen, '', docstyle_PYTHON3_doxygen.markers[1], TextPosition(30, 1)))
        self.assertEqual(list(DocBaseClass.extract(data, 'PYTHON3', 'doxygen')), expected)

    def test_DocBaseClass_extraction_PYTHON3_2(self):
        if False:
            print('Hello World!')
        data = ['\n', '""" documentation in single line  """\n', 'print(1)\n']
        docstyle_PYTHON3_default = DocstyleDefinition.load('PYTHON3', 'default')
        self.assertEqual(list(DocBaseClass.extract(data, 'PYTHON3', 'default')), [DocumentationComment(' documentation in single line  ', docstyle_PYTHON3_default, '', docstyle_PYTHON3_default.markers[0], TextPosition(2, 1))])

    def test_DocBaseClass_extraction_PYTHON3_3(self):
        if False:
            while True:
                i = 10
        data = ['## documentation in single line without return at end.']
        docstyle_PYTHON3_doxygen = DocstyleDefinition.load('PYTHON3', 'doxygen')
        self.assertEqual(list(DocBaseClass.extract(data, 'PYTHON3', 'doxygen')), [DocumentationComment(' documentation in single line without return at end.', docstyle_PYTHON3_doxygen, '', docstyle_PYTHON3_doxygen.markers[1], TextPosition(1, 1))])

    def test_DocBaseClass_extraction_PYTHON3_4(self):
        if False:
            for i in range(10):
                print('nop')
        data = ['\n', 'triple_quote_string_test = """\n', 'This is not a docstring\n', '"""\n']
        docstyle_PYTHON3_default = DocstyleDefinition.load('PYTHON3', 'default')
        self.assertEqual(list(DocBaseClass.extract(data, 'PYTHON3', 'default')), [])

    def test_DocBaseClass_extraction_PYTHON3_5(self):
        if False:
            while True:
                i = 10
        data = ['r"""\n', 'This is a raw docstring\n', '"""\n']
        docstyle_PYTHON3_default = DocstyleDefinition.load('PYTHON3', 'default')
        self.assertEqual(list(DocBaseClass.extract(data, 'PYTHON3', 'default')), [DocumentationComment('\nThis is a raw docstring\n', docstyle_PYTHON3_default, 'r', docstyle_PYTHON3_default.markers[0], TextPosition(1, 2))])

    def test_DocBaseClass_instantiate_padding_PYTHON3_6(self):
        if False:
            i = 10
            return i + 15
        data = ['def some_function:\n', '\n', '   """ documentation in single line """\n', '\n', '\n', 'print(1)']
        docstyle_PYTHON3_default = DocstyleDefinition.load('PYTHON3', 'default')
        for doc in DocBaseClass.extract(data, 'PYTHON3', 'default'):
            self.assertEqual([doc.top_padding, doc.bottom_padding], [1, 2])

    def test_DocBaseClass_instantiate_padding_PYTHON3_7(self):
        if False:
            for i in range(10):
                print('nop')
        data = ['class some_class:\n', '\n', '\n', '   """ documentation in single line """\n', '\n', '\n', 'print(1)\n']
        docstyle_PYTHON3_default = DocstyleDefinition.load('PYTHON3', 'default')
        for doc in DocBaseClass.extract(data, 'PYTHON3', 'default'):
            self.assertEqual([doc.top_padding, doc.bottom_padding], [2, 2])

    def test_DocBaseClass_instantiate_padding_inline_PYTHON3_8(self):
        if False:
            print('Hello World!')
        data = ['def some_function:\n', '\n', '   """\n', '   documentation in single line\n', '   """ # This is inline docstring', '\n', '\n', 'print(1)']
        docstyle_PYTHON3_default = DocstyleDefinition.load('PYTHON3', 'default')
        for doc in DocBaseClass.extract(data, 'PYTHON3', 'default'):
            self.assertEqual([doc.top_padding, doc.bottom_padding], [1, 0])

    def test_DocBaseClass_instantiate_padding_inline_PYTHON3_9(self):
        if False:
            while True:
                i = 10
        data = ['\n', '   """\n', '   documentation in single line\n', '   """ # This is inline docstring', '\n', '\n', 'print(1)']
        docstyle_PYTHON3_default = DocstyleDefinition.load('PYTHON3', 'default')
        for doc in DocBaseClass.extract(data, 'PYTHON3', 'default'):
            self.assertEqual([doc.top_padding, doc.bottom_padding], [0, 0])

    def test_DocBaseClass_instantiate_docstring_type_PYTHON3_10(self):
        if False:
            return 10
        data = ['class xyz:\n', '   """\n', '   This docstring is of docstring_type class\n', '   """\n']
        docstyle_PYTHON3_default = DocstyleDefinition.load('PYTHON3', 'default')
        for doc in DocBaseClass.extract(data, 'PYTHON3', 'default'):
            self.assertEqual(doc.docstring_type, 'class')

    def test_DocBaseClass_instantiate_docstring_type_PYTHON3_11(self):
        if False:
            for i in range(10):
                print('nop')
        data = ['def xyz:\n', '   """\n', '   This docstring is of docstring_type function\n', '   """\n']
        docstyle_PYTHON3_default = DocstyleDefinition.load('PYTHON3', 'default')
        for doc in DocBaseClass.extract(data, 'PYTHON3', 'default'):
            self.assertEqual(doc.docstring_type, 'function')

    def test_DocBaseClass_instantiate_docstring_type_PYTHON3_12(self):
        if False:
            for i in range(10):
                print('nop')
        data = ['\n', '   """\n', '   This docstring is of docstring_type others\n', '   """\n', '\n', 'print(1)']
        docstyle_PYTHON3_default = DocstyleDefinition.load('PYTHON3', 'default')
        for doc in DocBaseClass.extract(data, 'PYTHON3', 'default'):
            self.assertEqual(doc.docstring_type, 'others')

    def test_DocBaseClass_instantiate_docstring_type_PYTHON3_13(self):
        if False:
            for i in range(10):
                print('nop')
        data = ['## Documentation for a function.\n', '#\n', '#  More details.\n', 'def func():\n', '    pass\n']
        docstyle_PYTHON3_doxygen = DocstyleDefinition.load('PYTHON3', 'doxygen')
        for doc in DocBaseClass.extract(data, 'PYTHON3', 'doxygen'):
            self.assertEqual(doc.docstring_type, 'function')

    def test_DocBaseClass_instantiate_docstring_type_PYTHON3_14(self):
        if False:
            print('Hello World!')
        data = ['## Documentation for a class.\n', '#\n', '#  More details.\n', 'class PyClass:\n', '\n']
        docstyle_PYTHON3_doxygen = DocstyleDefinition.load('PYTHON3', 'doxygen')
        for doc in DocBaseClass.extract(data, 'PYTHON3', 'doxygen'):
            self.assertEqual(doc.docstring_type, 'class')

    def test_DocBaseClass_instantiate_docstring_type_PYTHON3_15(self):
        if False:
            print('Hello World!')
        data = ['def some_function():\n', '"""\n', 'documentation\n', '"""\n', 'class myPrivateClass:\n', '    pass']
        docstyle_PYTHON3_default = DocstyleDefinition.load('PYTHON3', 'default')
        for doc in DocBaseClass.extract(data, 'PYTHON3', 'default'):
            self.assertEqual(doc.docstring_type, 'function')

    def test_generate_diff(self):
        if False:
            print('Hello World!')
        data_old = ['\n', '""" documentation in single line  """\n']
        for doc_comment in DocBaseClass.extract(data_old, 'PYTHON3', 'default'):
            old_doc_comment = doc_comment
        old_range = TextRange.from_values(old_doc_comment.range.start.line, 1, old_doc_comment.range.end.line, old_doc_comment.range.end.column)
        data_new = ['\n', '"""\n documentation in single line\n"""\n']
        for doc_comment in DocBaseClass.extract(data_new, 'PYTHON3', 'default'):
            new_doc_comment = doc_comment
        diff = DocBaseClass.generate_diff(data_old, old_doc_comment, new_doc_comment)
        diff_expected = Diff(data_old)
        diff_expected.replace(old_range, new_doc_comment.assemble())
        self.assertEqual(diff, diff_expected)

    def test_DocBaseClass_process_documentation_not_implemented(self):
        if False:
            i = 10
            return i + 15
        test_object = DocBaseClass()
        self.assertRaises(NotImplementedError, test_object.process_documentation)

    def test_MalformedComment1_C(self):
        if False:
            for i in range(10):
                print('nop')
        data = ['/**\n', '* A doc-comment aborted in the middle of writing\n', "* This won't get parsed (hopefully...)\n"]
        expected = [dedent('            Please check the docstring for faulty markers. A starting\n            marker has been found, but no instance of DocComment is\n            returned.'), 0]
        for doc_comment in DocBaseClass.extract(data, 'C', 'doxygen'):
            self.assertEqual([doc_comment.message, doc_comment.line], expected)

    def test_MalformedComment2_CPP(self):
        if False:
            print('Hello World!')
        data = ['\n', '/** Aborts...\n']
        expected = [dedent('            Please check the docstring for faulty markers. A starting\n            marker has been found, but no instance of DocComment is\n            returned.'), 1]
        for doc_comment in DocBaseClass.extract(data, 'CPP', 'doxygen'):
            self.assertEqual([doc_comment.message, doc_comment.line], expected)

    def test_MalformedComment3_JAVA(self):
        if False:
            for i in range(10):
                print('nop')
        data = ['/**\n', '* Markers are faulty\n', '*/']
        expected = [dedent('            Please check the docstring for faulty markers. A starting\n            marker has been found, but no instance of DocComment is\n            returned.'), 0]
        for doc_comment in DocBaseClass.extract(data, 'JAVA', 'default'):
            self.assertEqual([doc_comment.message, doc_comment.line], expected)