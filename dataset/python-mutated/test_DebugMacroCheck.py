import inspect
import pathlib
import sys
import unittest
test_file = pathlib.Path(__file__)
sys.path.append(str(test_file.parent.parent))
import DebugMacroCheck
from os import linesep
from tests import DebugMacroDataSet
from tests import MacroTest
from typing import Callable, Tuple

class Meta_TestDebugMacroCheck(type):
    """
    Metaclass for debug macro test case class factory.
    """

    @classmethod
    def __prepare__(mcls, name, bases, **kwargs):
        if False:
            return 10
        'Returns the test case namespace for this class.'
        (candidate_macros, cls_ns, cnt) = ([], {}, 0)
        if 'category' in kwargs.keys():
            candidate_macros = [m for m in DebugMacroDataSet.DEBUG_MACROS if m.category == kwargs['category']]
        else:
            candidate_macros = DebugMacroDataSet.DEBUG_MACROS
        for (cnt, macro_test) in enumerate(candidate_macros):
            f_name = f'test_{macro_test.category}_{cnt}'
            t_desc = f'{macro_test!s}'
            cls_ns[f_name] = mcls.build_macro_test(macro_test, t_desc)
        return cls_ns

    def __new__(mcls, name, bases, ns, **kwargs):
        if False:
            i = 10
            return i + 15
        'Defined to prevent variable args from bubbling to the base class.'
        return super().__new__(mcls, name, bases, ns)

    def __init__(mcls, name, bases, ns, **kwargs):
        if False:
            while True:
                i = 10
        'Defined to prevent variable args from bubbling to the base class.'
        return super().__init__(name, bases, ns)

    @classmethod
    def build_macro_test(cls, macro_test: MacroTest.MacroTest, test_desc: str) -> Callable[[None], None]:
        if False:
            i = 10
            return i + 15
        'Returns a test function for this macro test data."\n\n        Args:\n            macro_test (MacroTest.MacroTest): The macro test class.\n\n            test_desc (str): A test description string.\n\n        Returns:\n            Callable[[None], None]: A test case function.\n        '

        def test_func(self):
            if False:
                print('Hello World!')
            act_result = cls.check_regex(macro_test.macro)
            self.assertCountEqual(act_result, macro_test.result, test_desc + f'{linesep}'.join(['', f'Actual Result:    {act_result}', '=' * 80, '']))
        return test_func

    @classmethod
    def check_regex(cls, source_str: str) -> Tuple[int, int, int]:
        if False:
            i = 10
            return i + 15
        'Returns the plugin result for the given macro string.\n\n        Args:\n            source_str (str): A string containing debug macros.\n\n        Returns:\n            Tuple[int, int, int]: A tuple of the number of formatting errors,\n            number of print specifiers, and number of arguments for the macros\n            given.\n        '
        return DebugMacroCheck.check_debug_macros(DebugMacroCheck.get_debug_macros(source_str), cls._get_function_name())

    @classmethod
    def _get_function_name(cls) -> str:
        if False:
            return 10
        'Returns the function name from one level of call depth.\n\n        Returns:\n            str: The caller function name.\n        '
        return 'function: ' + inspect.currentframe().f_back.f_code.co_name

class Test_NoSpecifierNoArgument(unittest.TestCase, metaclass=Meta_TestDebugMacroCheck, category='no_specifier_no_argument_macro_test'):
    pass

class Test_EqualSpecifierEqualArgument(unittest.TestCase, metaclass=Meta_TestDebugMacroCheck, category='equal_specifier_equal_argument_macro_test'):
    pass

class Test_MoreSpecifiersThanArguments(unittest.TestCase, metaclass=Meta_TestDebugMacroCheck, category='more_specifiers_than_arguments_macro_test'):
    pass

class Test_LessSpecifiersThanArguments(unittest.TestCase, metaclass=Meta_TestDebugMacroCheck, category='less_specifiers_than_arguments_macro_test'):
    pass

class Test_IgnoredSpecifiers(unittest.TestCase, metaclass=Meta_TestDebugMacroCheck, category='ignored_specifiers_macro_test'):
    pass

class Test_SpecialParsingMacroTest(unittest.TestCase, metaclass=Meta_TestDebugMacroCheck, category='special_parsing_macro_test'):
    pass

class Test_CodeSnippetMacroTest(unittest.TestCase, metaclass=Meta_TestDebugMacroCheck, category='code_snippet_macro_test'):
    pass
if __name__ == '__main__':
    unittest.main()