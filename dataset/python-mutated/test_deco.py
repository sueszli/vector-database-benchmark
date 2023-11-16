import unittest
from robot.api.deco import keyword, library
from robot.utils.asserts import assert_equal, assert_false

class TestKeywordName(unittest.TestCase):

    def test_give_name_to_function(self):
        if False:
            print('Hello World!')

        @keyword('Given name')
        def func():
            if False:
                print('Hello World!')
            pass
        assert_equal(func.robot_name, 'Given name')

    def test_give_name_to_method(self):
        if False:
            for i in range(10):
                print('nop')

        class Class:

            @keyword('Given name')
            def method(self):
                if False:
                    while True:
                        i = 10
                pass
        assert_equal(Class.method.robot_name, 'Given name')

    def test_no_name(self):
        if False:
            for i in range(10):
                print('nop')

        @keyword()
        def func():
            if False:
                while True:
                    i = 10
            pass
        assert_equal(func.robot_name, None)

    def test_no_name_nor_parens(self):
        if False:
            print('Hello World!')

        @keyword
        def func():
            if False:
                i = 10
                return i + 15
            pass
        assert_equal(func.robot_name, None)

class TestLibrary(unittest.TestCase):

    def test_auto_keywords_is_disabled_by_default(self):
        if False:
            print('Hello World!')

        @library
        class lib1:
            pass

        @library()
        class lib2:
            pass
        self._validate_lib(lib1)
        self._validate_lib(lib2)

    def test_auto_keywords_can_be_enabled(self):
        if False:
            print('Hello World!')

        @library(auto_keywords=False)
        class lib:
            pass
        self._validate_lib(lib, auto_keywords=False)

    def test_other_options(self):
        if False:
            return 10

        @library('GLOBAL', version='v', doc_format='HTML', listener='xx')
        class lib:
            pass
        self._validate_lib(lib, 'GLOBAL', 'v', 'HTML', 'xx')

    def test_override_class_level_attributes(self):
        if False:
            for i in range(10):
                print('nop')

        @library(doc_format='HTML', listener='xx', scope='GLOBAL', version='v', auto_keywords=True)
        class lib:
            ROBOT_LIBRARY_SCOPE = 'override'
            ROBOT_LIBRARY_VERSION = 'override'
            ROBOT_LIBRARY_DOC_FORMAT = 'override'
            ROBOT_LIBRARY_LISTENER = 'override'
            ROBOT_AUTO_KEYWORDS = 'override'
        self._validate_lib(lib, 'GLOBAL', 'v', 'HTML', 'xx', True)

    def _validate_lib(self, lib, scope=None, version=None, doc_format=None, listener=None, auto_keywords=False):
        if False:
            i = 10
            return i + 15
        self._validate_attr(lib, 'ROBOT_LIBRARY_SCOPE', scope)
        self._validate_attr(lib, 'ROBOT_LIBRARY_VERSION', version)
        self._validate_attr(lib, 'ROBOT_LIBRARY_DOC_FORMAT', doc_format)
        self._validate_attr(lib, 'ROBOT_LIBRARY_LISTENER', listener)
        self._validate_attr(lib, 'ROBOT_AUTO_KEYWORDS', auto_keywords)

    def _validate_attr(self, lib, attr, value):
        if False:
            i = 10
            return i + 15
        if value is None:
            assert_false(hasattr(lib, attr))
        else:
            assert_equal(getattr(lib, attr), value)
if __name__ == '__main__':
    unittest.main()