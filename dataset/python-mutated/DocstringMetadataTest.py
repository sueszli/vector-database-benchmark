import unittest
from coalib.settings.DocstringMetadata import DocstringMetadata
from collections import OrderedDict

class DocstringMetadataTest(unittest.TestCase):

    def test_from_docstring(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_from_docstring_dataset('')
        self.check_from_docstring_dataset(' description only ', desc='description only')
        self.check_from_docstring_dataset(' :param test:  test description ', param_dict={'test': 'test description'})
        self.check_from_docstring_dataset(' @param test:  test description ', param_dict={'test': 'test description'})
        self.check_from_docstring_dataset(' :return: something ', retval_desc='something')
        self.check_from_docstring_dataset(' @return: something ', retval_desc='something')
        self.check_from_docstring_dataset('\n        Main description\n\n        @param p1: this is\n\n        a multiline desc for p1\n\n        :param p2: p2 description\n\n        @return: retval description\n        :return: retval description\n        override\n        ', desc='Main description', param_dict={'p1': 'this is\na multiline desc for p1\n', 'p2': 'p2 description\n'}, retval_desc='retval description override')

    def test_str(self):
        if False:
            print('Hello World!')
        uut = DocstringMetadata.from_docstring('\n            Description of something. No params.\n            ')
        self.assertEqual(str(uut), 'Description of something. No params.')
        uut = DocstringMetadata.from_docstring('\n            Description of something with params.\n\n            :param x: Imagine something.\n            :param y: x^2\n            ')
        self.assertEqual(str(uut), 'Description of something with params.')

    def test_unneeded_docstring_space(self):
        if False:
            for i in range(10):
                print('nop')
        uut = DocstringMetadata.from_docstring('\n            This is a description about some bear which does some amazing\n            things. This is a multiline description for this testcase.\n\n            :param language:\n                The programming language.\n            :param coalang_dir:\n                External directory for coalang file.\n            ')
        expected_output = OrderedDict([('language', 'The programming language.'), ('coalang_dir', 'External directory for coalang file.')])
        self.assertEqual(uut.param_dict, expected_output)

    def check_from_docstring_dataset(self, docstring, desc='', param_dict=None, retval_desc=''):
        if False:
            i = 10
            return i + 15
        param_dict = param_dict or {}
        self.assertIsInstance(docstring, str, 'docstring needs to be a string for this test.')
        doc_comment = DocstringMetadata.from_docstring(docstring)
        self.assertEqual(doc_comment.desc, desc)
        self.assertEqual(doc_comment.param_dict, param_dict)
        self.assertEqual(doc_comment.retval_desc, retval_desc)