import pickle
import unittest
from coalib.bearlib.languages.Language import Language, LanguageMeta

class LanguageTest(unittest.TestCase):

    def test_class__dir__(self):
        if False:
            print('Hello World!')
        assert set(dir(Language)) == {lang.__name__ for lang in LanguageMeta.all}.union(type.__dir__(Language))

    def test_pickle_ability(self):
        if False:
            return 10
        cpp = Language['CPP']
        cpp_str = pickle.dumps(cpp)
        cpp_unpickled = pickle.loads(cpp_str)
        self.assertEqual(str(cpp), str(cpp_unpickled))

    def test_contains_method(self):
        if False:
            print('Hello World!')
        self.assertTrue('py' in Language[Language.Python])
        self.assertTrue('python' in Language[Language.Python == 3])
        self.assertTrue('python' in Language['python 3'])
        self.assertFalse('py 2' in Language['py 3'])
        self.assertFalse('py 2.7, 3.4' in Language['py 3'])

class LanguageAttributeErrorTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.lang_cpp = Language['CPP']
        self.lang_unknown = Language['Unknown']

    def tearDown(self):
        if False:
            print('Hello World!')
        pass

    def test_invalid_attribute(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(AttributeError, 'not a valid attribute'):
            self.lang_cpp.not_an_attribute

    def test_attribute_list_empy(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(AttributeError, 'no available attribute'):
            self.lang_unknown.not_an_attribute