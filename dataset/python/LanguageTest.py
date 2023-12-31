import pickle
import unittest

from coalib.bearlib.languages.Language import Language, LanguageMeta


class LanguageTest(unittest.TestCase):

    def test_class__dir__(self):
        assert set(dir(Language)) == {
            lang.__name__ for lang in LanguageMeta.all
        }.union(type.__dir__(Language))

    def test_pickle_ability(self):
        cpp = Language['CPP']
        cpp_str = pickle.dumps(cpp)
        cpp_unpickled = pickle.loads(cpp_str)
        self.assertEqual(str(cpp), str(cpp_unpickled))

    def test_contains_method(self):
        # Test alias
        self.assertTrue('py' in Language[Language.Python])
        # Test version
        self.assertTrue('python' in Language[Language.Python == 3])
        # Test string parse
        self.assertTrue('python' in Language['python 3'])
        # Test version exclusion
        self.assertFalse('py 2' in Language['py 3'])
        # More complex version exclusion test
        self.assertFalse('py 2.7, 3.4' in Language['py 3'])


class LanguageAttributeErrorTest(unittest.TestCase):

    def setUp(self):
        self.lang_cpp = Language['CPP']
        self.lang_unknown = Language['Unknown']

    def tearDown(self):
        pass

    def test_invalid_attribute(self):
        with self.assertRaisesRegex(AttributeError, 'not a valid attribute'):
            self.lang_cpp.not_an_attribute

    def test_attribute_list_empy(self):
        with self.assertRaisesRegex(AttributeError, 'no available attribute'):
            self.lang_unknown.not_an_attribute
