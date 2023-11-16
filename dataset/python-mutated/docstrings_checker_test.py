"""Unit tests for scripts/docstrings_checker."""
from __future__ import annotations
from core.tests import test_utils
from . import docstrings_checker
import astroid
from pylint.checkers import utils

class DocstringsCheckerTest(test_utils.GenericTestBase):
    """Class for testing the docstrings_checker script."""

    def test_space_indentation(self) -> None:
        if False:
            i = 10
            return i + 15
        sample_string = '     This is a sample string.'
        self.assertEqual(docstrings_checker.space_indentation(sample_string), 5)

    def test_get_setters_property_name_with_setter(self) -> None:
        if False:
            while True:
                i = 10
        setter_node = astroid.extract_node('\n        @test.setter\n        def func():\n            pass\n        ')
        property_name = docstrings_checker.get_setters_property_name(setter_node)
        self.assertEqual(property_name, 'test')

    def test_get_setters_property_name_without_setter(self) -> None:
        if False:
            print('Hello World!')
        none_node = astroid.extract_node('\n        @attribute\n        def func():\n            pass\n        ')
        none_return = docstrings_checker.get_setters_property_name(none_node)
        self.assertEqual(none_return, None)

    def test_get_setters_property_with_setter_and_property(self) -> None:
        if False:
            i = 10
            return i + 15
        node = astroid.extract_node('\n        class TestClass():\n            @test.setter\n            @property\n            def func():\n                pass\n        ')
        temp = node.getattr('func')
        setter_property = docstrings_checker.get_setters_property(temp[0])
        self.assertEqual(isinstance(setter_property, astroid.FunctionDef), True)

    def test_get_setters_property_with_setter_no_property(self) -> None:
        if False:
            i = 10
            return i + 15
        testnode2 = astroid.extract_node('\n        class TestClass():\n            @test.setter\n            def func():\n                pass\n        ')
        temp = testnode2.getattr('func')
        setter_property = docstrings_checker.get_setters_property(temp[0])
        self.assertEqual(setter_property, None)

    def test_get_setters_property_no_class(self) -> None:
        if False:
            while True:
                i = 10
        testnode3 = astroid.extract_node('\n        @test.setter\n        def func():\n            pass\n        ')
        setter_property = docstrings_checker.get_setters_property(testnode3)
        self.assertEqual(setter_property, None)

    def test_get_setters_property_no_setter_no_property(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        testnode4 = astroid.extract_node('\n        class TestClass():\n            def func():\n                pass\n        ')
        temp = testnode4.getattr('func')
        setter_property = docstrings_checker.get_setters_property(temp[0])
        self.assertEqual(setter_property, None)

    def test_returns_something_with_value_retur(self) -> None:
        if False:
            while True:
                i = 10
        return_node = astroid.extract_node('\n        return True\n        ')
        self.assertEqual(docstrings_checker.returns_something(return_node), True)

    def test_returns_something_with_none_return(self) -> None:
        if False:
            return 10
        return_none_node = astroid.extract_node('\n        return None\n        ')
        self.assertEqual(docstrings_checker.returns_something(return_none_node), False)

    def test_returns_something_with_empty_return(self) -> None:
        if False:
            print('Hello World!')
        none_return_node = astroid.extract_node('\n        return\n        ')
        self.assertEqual(docstrings_checker.returns_something(none_return_node), False)

    def test_possible_exc_types_with_valid_name(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise_node = astroid.extract_node('\n        def func():\n            raise IndexError #@\n        ')
        exceptions = docstrings_checker.possible_exc_types(raise_node)
        self.assertEqual(exceptions, set(['IndexError']))

    def test_possible_exc_types_with_invalid_name(self) -> None:
        if False:
            print('Hello World!')
        raise_node = astroid.extract_node('\n        def func():\n            raise AInvalidError #@\n        ')
        exceptions = docstrings_checker.possible_exc_types(raise_node)
        self.assertEqual(exceptions, set([]))

    def test_possible_exc_types_with_function_call_no_return(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise_node = astroid.extract_node('\n        def testFunc():\n            pass\n\n        def func():\n            raise testFunc() #@\n        ')
        excpetions = docstrings_checker.possible_exc_types(raise_node)
        self.assertEqual(excpetions, set([]))

    def test_possible_exc_types_with_function_call_valid_errors(self) -> None:
        if False:
            i = 10
            return i + 15
        raise_node = astroid.extract_node('\n        def testFunc():\n            if True:\n                return IndexError\n            else:\n                return ValueError\n\n        def func():\n            raise testFunc() #@\n        ')
        excpetions = docstrings_checker.possible_exc_types(raise_node)
        self.assertEqual(excpetions, set(['IndexError', 'ValueError']))

    def test_possible_exc_types_with_function_call_invalid_error(self) -> None:
        if False:
            return 10
        raise_node = astroid.extract_node('\n        def testFunc():\n            return AInvalidError\n\n        def func():\n            raise testFunc() #@\n        ')
        excpetions = docstrings_checker.possible_exc_types(raise_node)
        self.assertEqual(excpetions, set([]))

    def test_possible_exc_types_with_return_out_of_frame(self) -> None:
        if False:
            return 10
        raise_node = astroid.extract_node('\n        def testFunc():\n            def inner():\n                return IndexError\n\n            pass\n\n        def func():\n            raise testFunc() #@\n        ')
        exceptions = docstrings_checker.possible_exc_types(raise_node)
        self.assertEqual(exceptions, set([]))

    def test_possible_exc_types_with_undefined_function_call(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise_node = astroid.extract_node('\n        def func():\n            raise testFunc() #@\n        ')
        excpetions = docstrings_checker.possible_exc_types(raise_node)
        self.assertEqual(excpetions, set([]))

    def test_possible_exc_types_with_constaint_raise(self) -> None:
        if False:
            print('Hello World!')
        raise_node = astroid.extract_node('\n        def func():\n            raise True #@\n        ')
        exceptions = docstrings_checker.possible_exc_types(raise_node)
        self.assertEqual(exceptions, set([]))

    def test_possible_exc_types_with_inference_error(self) -> None:
        if False:
            return 10
        raise_node = astroid.extract_node("\n        def func():\n            raise Exception('An exception.') #@\n        ")
        node_ignores_exception_swap = self.swap(utils, 'node_ignores_exception', lambda _, __: (_ for _ in ()).throw(astroid.InferenceError()))
        with node_ignores_exception_swap:
            exceptions = docstrings_checker.possible_exc_types(raise_node)
        self.assertEqual(exceptions, set([]))

    def test_possible_exc_types_with_exception_message(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise_node = astroid.extract_node('\n        def func():\n            """Function to test raising exceptions."""\n            raise Exception(\'An exception.\') #@\n        ')
        exceptions = docstrings_checker.possible_exc_types(raise_node)
        self.assertEqual(exceptions, set(['Exception']))

    def test_possible_exc_types_with_no_exception(self) -> None:
        if False:
            return 10
        raise_node = astroid.extract_node('\n        def func():\n            """Function to test raising exceptions."""\n            raise #@\n        ')
        exceptions = docstrings_checker.possible_exc_types(raise_node)
        self.assertEqual(exceptions, set([]))

    def test_possible_exc_types_with_exception_inside_function(self) -> None:
        if False:
            while True:
                i = 10
        raise_node = astroid.extract_node("\n        def func():\n            try:\n                raise Exception('An exception.')\n            except Exception:\n                raise #@\n        ")
        exceptions = docstrings_checker.possible_exc_types(raise_node)
        self.assertEqual(exceptions, set(['Exception']))

    def test_docstringify_with_valid_docstring(self) -> None:
        if False:
            while True:
                i = 10
        valid_docstring = astroid.extract_node("\n        def func():\n            '''Docstring that is correctly formated\n                according to the Google Python Style Guide.\n\n            Args:\n                test_value: bool. Just a test argument.\n            '''\n            pass\n            ").doc_node
        is_valid = isinstance(docstrings_checker.docstringify(valid_docstring), docstrings_checker.GoogleDocstring)
        self.assertEqual(is_valid, True)

    def test_docstringify_with_invalid_docstring(self) -> None:
        if False:
            while True:
                i = 10
        invalid_docstring = astroid.extract_node("\n        def func():\n            '''Docstring that is incorrectly formated\n                according to the Google Python Style Guide.\n            '''\n            pass\n            ").doc_node
        is_valid = isinstance(docstrings_checker.docstringify(invalid_docstring), docstrings_checker.GoogleDocstring)
        self.assertEqual(is_valid, False)