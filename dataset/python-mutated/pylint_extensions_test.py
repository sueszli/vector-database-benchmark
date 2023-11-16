"""Unit tests for scripts/pylint_extensions."""
from __future__ import annotations
import tempfile
import unittest
from core import utils
from . import pylint_extensions
import astroid
from pylint import interfaces
from pylint import testutils
from pylint import utils as pylint_utils

class HangingIndentCheckerTests(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        self.checker_test_object = testutils.CheckerTestCase()
        self.checker_test_object.CHECKER_CLASS = pylint_extensions.HangingIndentChecker
        self.checker_test_object.setup_method()

    def test_no_break_after_hanging_indentation(self) -> None:
        if False:
            i = 10
            return i + 15
        node_break_after_hanging_indent = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u"self.post_json('/ml/\\trainedclassifierhandler',\n                self.payload, expect_errors=True, expected_status_int=401)\n                if (a > 1 and\n                        b > 2):\n                ")
        node_break_after_hanging_indent.file = filename
        node_break_after_hanging_indent.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_break_after_hanging_indent))
        message = testutils.MessageTest(msg_id='no-break-after-hanging-indent', line=1)
        with self.checker_test_object.assertAddsMessages(message):
            temp_file.close()

    def test_no_break_after_hanging_indentation_with_comment(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        node_break_after_hanging_indent = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u"self.post_json('/ml/\\trainedclassifierhandler',\n                self.payload, expect_errors=True, expected_status_int=401)\n\n                if (a > 1 and\n                        b > 2):  # pylint: disable=invalid-name\n                ")
        node_break_after_hanging_indent.file = filename
        node_break_after_hanging_indent.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_break_after_hanging_indent))
        message = testutils.MessageTest(msg_id='no-break-after-hanging-indent', line=1)
        with self.checker_test_object.assertAddsMessages(message):
            temp_file.close()

    def test_break_after_hanging_indentation(self) -> None:
        if False:
            i = 10
            return i + 15
        node_with_no_error_message = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'"""Some multiline\n                docstring.\n                """\n                # Load JSON.\n                master_translation_dict = json.loads(\n               pylint_utils.get_file_contents(os.path.join(\n                os.getcwd(), \'assets\', \'i18n\', \'en.json\')))\n                ')
        node_with_no_error_message.file = filename
        node_with_no_error_message.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_with_no_error_message))
        with self.checker_test_object.assertNoMessages():
            temp_file.close()

    def test_hanging_indentation_with_a_comment_after_bracket(self) -> None:
        if False:
            return 10
        node_with_no_error_message = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u"self.post_json(  # Random comment\n                '(',\n                self.payload, expect_errors=True, expected_status_int=401)")
        node_with_no_error_message.file = filename
        node_with_no_error_message.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_with_no_error_message))
        with self.checker_test_object.assertNoMessages():
            temp_file.close()

    def test_hanging_indentation_with_a_comment_after_two_or_more_bracket(self) -> None:
        if False:
            print('Hello World!')
        node_with_no_error_message = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u"self.post_json(func(  # Random comment\n                '(',\n                self.payload, expect_errors=True, expected_status_int=401))")
        node_with_no_error_message.file = filename
        node_with_no_error_message.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_with_no_error_message))
        with self.checker_test_object.assertNoMessages():
            temp_file.close()

    def test_hanging_indentation_with_a_comment_after_square_bracket(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        node_with_no_error_message = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u"self.post_json([  # Random comment\n                '(',\n                '', '', ''])")
        node_with_no_error_message.file = filename
        node_with_no_error_message.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_with_no_error_message))
        with self.checker_test_object.assertNoMessages():
            temp_file.close()

    def test_hanging_indentation_with_a_if_statement_before(self) -> None:
        if False:
            return 10
        node_with_no_error_message = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u"\n                if 5 > 7:\n                    self.post_json([\n                    '(',\n                    '', '', ''])\n\n                def func(arg1,\n                    arg2, arg3):\n                    a = 2 / 2")
        node_with_no_error_message.file = filename
        node_with_no_error_message.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_with_no_error_message))
        message = testutils.MessageTest(msg_id='no-break-after-hanging-indent', line=7)
        with self.checker_test_object.assertAddsMessages(message):
            temp_file.close()

class DocstringParameterCheckerTests(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        self.checker_test_object = testutils.CheckerTestCase()
        self.checker_test_object.CHECKER_CLASS = pylint_extensions.DocstringParameterChecker
        self.checker_test_object.setup_method()

    def test_no_newline_below_class_docstring(self) -> None:
        if False:
            while True:
                i = 10
        node_no_newline_below_class_docstring = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                    class ClassName(dummy_class):\n                        """This is a docstring."""\n                        a = 1 + 2\n                ')
        node_no_newline_below_class_docstring.file = filename
        node_no_newline_below_class_docstring.path = filename
        self.checker_test_object.checker.visit_classdef(node_no_newline_below_class_docstring)
        message = testutils.MessageTest(msg_id='newline-below-class-docstring', node=node_no_newline_below_class_docstring)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            temp_file.close()

    def test_excessive_newline_below_class_docstring(self) -> None:
        if False:
            i = 10
            return i + 15
        node_excessive_newline_below_class_docstring = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                    class ClassName(dummy_class):\n                        """This is a docstring."""\n\n\n                        a = 1 + 2\n                ')
        node_excessive_newline_below_class_docstring.file = filename
        node_excessive_newline_below_class_docstring.path = filename
        self.checker_test_object.checker.visit_classdef(node_excessive_newline_below_class_docstring)
        message = testutils.MessageTest(msg_id='newline-below-class-docstring', node=node_excessive_newline_below_class_docstring)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            temp_file.close()

    def test_inline_comment_after_class_docstring(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        node_inline_comment_after_class_docstring = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                    class ClassName(dummy_class):\n                        """This is a docstring."""\n                        # This is a comment.\n                        def func():\n                            a = 1 + 2\n                ')
        node_inline_comment_after_class_docstring.file = filename
        node_inline_comment_after_class_docstring.path = filename
        self.checker_test_object.checker.visit_classdef(node_inline_comment_after_class_docstring)
        message = testutils.MessageTest(msg_id='newline-below-class-docstring', node=node_inline_comment_after_class_docstring)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            temp_file.close()

    def test_multiline_class_argument_with_incorrect_style(self) -> None:
        if False:
            return 10
        node_multiline_class_argument_with_incorrect_style = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                    class ClassName(\n                            dummy_class):\n                        """This is a docstring."""\n                        a = 1 + 2\n                ')
        node_multiline_class_argument_with_incorrect_style.file = filename
        node_multiline_class_argument_with_incorrect_style.path = filename
        self.checker_test_object.checker.visit_classdef(node_multiline_class_argument_with_incorrect_style)
        message = testutils.MessageTest(msg_id='newline-below-class-docstring', node=node_multiline_class_argument_with_incorrect_style)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            temp_file.close()

    def test_multiline_class_argument_with_correct_style(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        node_multiline_class_argument_with_correct_style = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                    class ClassName(\n                            dummy_class):\n                        """This is a docstring."""\n\n                        a = 1 + 2\n                ')
        node_multiline_class_argument_with_correct_style.file = filename
        node_multiline_class_argument_with_correct_style.path = filename
        self.checker_test_object.checker.visit_classdef(node_multiline_class_argument_with_correct_style)
        with self.checker_test_object.assertNoMessages():
            temp_file.close()

    def test_single_newline_below_class_docstring(self) -> None:
        if False:
            print('Hello World!')
        node_with_no_error_message = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                    class ClassName(dummy_class):\n                        """This is a multiline docstring."""\n\n                        a = 1 + 2\n                ')
        node_with_no_error_message.file = filename
        node_with_no_error_message.path = filename
        self.checker_test_object.checker.visit_classdef(node_with_no_error_message)
        with self.checker_test_object.assertNoMessages():
            temp_file.close()

    def test_class_with_no_docstring(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        node_class_with_no_docstring = astroid.scoped_nodes.Module(name='test', doc=None)
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                    class ClassName(dummy_class):\n                        a = 1 + 2\n                ')
        node_class_with_no_docstring.file = filename
        node_class_with_no_docstring.path = filename
        self.checker_test_object.checker.visit_classdef(node_class_with_no_docstring)
        with self.checker_test_object.assertNoMessages():
            temp_file.close()

    def test_newline_before_docstring_with_correct_style(self) -> None:
        if False:
            return 10
        node_newline_before_docstring_with_correct_style = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                    class ClassName(dummy_class):\n\n                        """This is a multiline docstring."""\n\n                        a = 1 + 2\n                ')
        node_newline_before_docstring_with_correct_style.file = filename
        node_newline_before_docstring_with_correct_style.path = filename
        self.checker_test_object.checker.visit_classdef(node_newline_before_docstring_with_correct_style)
        with self.checker_test_object.assertNoMessages():
            temp_file.close()

    def test_newline_before_docstring_with_incorrect_style(self) -> None:
        if False:
            while True:
                i = 10
        node_newline_before_docstring_with_incorrect_style = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                    class ClassName(dummy_class):\n\n                        """This is a multiline docstring."""\n                        a = 1 + 2\n                ')
        node_newline_before_docstring_with_incorrect_style.file = filename
        node_newline_before_docstring_with_incorrect_style.path = filename
        self.checker_test_object.checker.visit_classdef(node_newline_before_docstring_with_incorrect_style)
        message = testutils.MessageTest(msg_id='newline-below-class-docstring', node=node_newline_before_docstring_with_incorrect_style)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            temp_file.close()

    def test_malformed_args_section(self) -> None:
        if False:
            i = 10
            return i + 15
        node_malformed_args_section = astroid.extract_node(u'def func(arg): #@\n                """Does nothing.\n\n                Args:\n                    arg: Argument description.\n                """\n                a = True\n        ')
        message = testutils.MessageTest(msg_id='malformed-args-section', node=node_malformed_args_section)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(node_malformed_args_section)

    def test_malformed_returns_section(self) -> None:
        if False:
            return 10
        node_malformed_returns_section = astroid.extract_node(u'def func(): #@\n                """Return True.\n\n                Returns:\n                    arg: Argument description.\n                """\n                return True\n        ')
        message = testutils.MessageTest(msg_id='malformed-returns-section', node=node_malformed_returns_section)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(node_malformed_returns_section)

    def test_malformed_yields_section(self) -> None:
        if False:
            print('Hello World!')
        node_malformed_yields_section = astroid.extract_node(u'def func(): #@\n                """Yield true.\n\n                Yields:\n                    yields: Argument description.\n                """\n                yield True\n        ')
        message = testutils.MessageTest(msg_id='malformed-yields-section', node=node_malformed_yields_section)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(node_malformed_yields_section)

    def test_malformed_raises_section(self) -> None:
        if False:
            i = 10
            return i + 15
        node_malformed_raises_section = astroid.extract_node(u'def func(): #@\n                """Raise an exception.\n\n                Raises:\n                    Exception: Argument description.\n                """\n                raise Exception()\n        ')
        message = testutils.MessageTest(msg_id='malformed-raises-section', node=node_malformed_raises_section)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(node_malformed_raises_section)

    def test_malformed_args_argument(self) -> None:
        if False:
            while True:
                i = 10
        node_malformed_args_argument = astroid.extract_node(u'def func(*args): #@\n                """Does nothing.\n\n                Args:\n                    *args: int. Argument description.\n                """\n                a = True\n        ')
        message = testutils.MessageTest(msg_id='malformed-args-argument', node=node_malformed_args_argument)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(node_malformed_args_argument)

    def test_well_formated_args_argument(self) -> None:
        if False:
            print('Hello World!')
        node_with_no_error_message = astroid.extract_node(u'def func(*args): #@\n                """Does nothing.\n\n                Args:\n                    *args: list(*). Description.\n                """\n                a = True\n        ')
        with self.checker_test_object.assertAddsMessages():
            self.checker_test_object.checker.visit_functiondef(node_with_no_error_message)

    def test_well_formated_args_section(self) -> None:
        if False:
            i = 10
            return i + 15
        node_with_no_error_message = astroid.extract_node(u'def func(arg): #@\n                """Does nothing.\n\n                Args:\n                    arg: argument. Description.\n                """\n                a = True\n        ')
        with self.checker_test_object.assertAddsMessages():
            self.checker_test_object.checker.visit_functiondef(node_with_no_error_message)

    def test_well_formated_returns_section(self) -> None:
        if False:
            while True:
                i = 10
        node_with_no_error_message = astroid.extract_node(u'def func(): #@\n                """Does nothing.\n\n                Returns:\n                    int. Argument escription.\n                """\n                return args\n        ')
        with self.checker_test_object.assertAddsMessages():
            self.checker_test_object.checker.visit_functiondef(node_with_no_error_message)

    def test_well_formated_yields_section(self) -> None:
        if False:
            return 10
        node_with_no_error_message = astroid.extract_node(u'def func(): #@\n                """Does nothing.\n\n                Yields:\n                    arg. Argument description.\n                """\n                yield args\n        ')
        with self.checker_test_object.assertAddsMessages():
            self.checker_test_object.checker.visit_functiondef(node_with_no_error_message)

    def test_space_after_docstring(self) -> None:
        if False:
            while True:
                i = 10
        node_space_after_docstring = astroid.extract_node(u'def func():\n                    """ Hello world."""\n                    Something\n        ')
        message = testutils.MessageTest(msg_id='space-after-triple-quote', node=node_space_after_docstring)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(node_space_after_docstring)

    def test_two_lines_empty_docstring_raise_correct_message(self) -> None:
        if False:
            i = 10
            return i + 15
        node_with_docstring = astroid.extract_node(u'def func():\n                    """\n                    """\n                    pass\n        ')
        message = testutils.MessageTest(msg_id='single-line-docstring-span-two-lines', node=node_with_docstring)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(node_with_docstring)

    def test_single_line_docstring_span_two_lines(self) -> None:
        if False:
            while True:
                i = 10
        node_single_line_docstring_span_two_lines = astroid.extract_node(u'def func(): #@\n                    """This is a docstring.\n                    """\n                    Something\n        ')
        message = testutils.MessageTest(msg_id='single-line-docstring-span-two-lines', node=node_single_line_docstring_span_two_lines)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(node_single_line_docstring_span_two_lines)

    def test_no_period_at_end(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        node_no_period_at_end = astroid.extract_node(u'def func(): #@\n                    """This is a docstring"""\n                    Something\n        ')
        message = testutils.MessageTest(msg_id='no-period-used', node=node_no_period_at_end)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(node_no_period_at_end)

    def test_empty_line_before_end_of_docstring(self) -> None:
        if False:
            while True:
                i = 10
        node_empty_line_before_end = astroid.extract_node(u'def func(): #@\n                    """This is a docstring.\n\n                    """\n                    Something\n        ')
        message = testutils.MessageTest(msg_id='empty-line-before-end', node=node_empty_line_before_end)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(node_empty_line_before_end)

    def test_no_period_at_end_of_a_multiline_docstring(self) -> None:
        if False:
            print('Hello World!')
        node_no_period_at_end = astroid.extract_node(u'def func(arg): #@\n                    """This is a docstring.\n\n                        Args:\n                            arg: variable. Desciption\n                    """\n                    Something\n        ')
        no_period_at_end_message = testutils.MessageTest(msg_id='no-period-used', node=node_no_period_at_end)
        malformed_args_message = testutils.MessageTest(msg_id='malformed-args-section', node=node_no_period_at_end)
        with self.checker_test_object.assertAddsMessages(no_period_at_end_message, malformed_args_message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(node_no_period_at_end)

    def test_no_newline_at_end_of_multi_line_docstring(self) -> None:
        if False:
            i = 10
            return i + 15
        node_no_newline_at_end = astroid.extract_node(u'def func(arg): #@\n                    """This is a docstring.\n\n                        Args:\n                            arg: variable. Description."""\n                    Something\n        ')
        message = testutils.MessageTest(msg_id='no-newline-used-at-end', node=node_no_newline_at_end)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(node_no_newline_at_end)

    def test_no_newline_above_args(self) -> None:
        if False:
            i = 10
            return i + 15
        node_single_newline_above_args = astroid.extract_node(u'def func(arg): #@\n                """Do something.\n                Args:\n                    arg: argument. Description.\n                """\n        ')
        message = testutils.MessageTest(msg_id='single-space-above-args', node=node_single_newline_above_args)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(node_single_newline_above_args)

    def test_no_newline_above_raises(self) -> None:
        if False:
            i = 10
            return i + 15
        node_single_newline_above_raises = astroid.extract_node(u'def func(): #@\n                    """Raises exception.\n                    Raises:\n                        raises_exception. Description.\n                    """\n                    raise exception\n        ')
        message = testutils.MessageTest(msg_id='single-space-above-raises', node=node_single_newline_above_raises)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(node_single_newline_above_raises)

    def test_no_newline_above_return(self) -> None:
        if False:
            while True:
                i = 10
        node_with_no_space_above_return = astroid.extract_node(u'def func(): #@\n                """Returns something.\n                Returns:\n                    returns_something. Description.\n                """\n                return something\n        ')
        message = testutils.MessageTest(msg_id='single-space-above-returns', node=node_with_no_space_above_return)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(node_with_no_space_above_return)

    def test_varying_combination_of_newline_above_args(self) -> None:
        if False:
            print('Hello World!')
        node_newline_above_args_raises = astroid.extract_node(u'def func(arg): #@\n                """Raises exception.\n\n                Args:\n                    arg: argument. Description.\n                Raises:\n                    raises_something. Description.\n                """\n                raise exception\n        ')
        message = testutils.MessageTest(msg_id='single-space-above-raises', node=node_newline_above_args_raises)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(node_newline_above_args_raises)
        node_newline_above_args_returns = astroid.extract_node(u'def func(arg): #@\n                """Returns Something.\n\n                Args:\n                    arg: argument. Description.\n                Returns:\n                    returns_something. Description.\n                """\n                return something\n        ')
        message = testutils.MessageTest(msg_id='single-space-above-returns', node=node_newline_above_args_returns)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(node_newline_above_args_returns)
        node_newline_above_returns_raises = astroid.extract_node(u'def func(): #@\n                """Do something.\n\n\n\n                Raises:\n                    raises_exception. Description.\n\n                Returns:\n                    returns_something. Description.\n                """\n                raise something\n                return something\n        ')
        message = testutils.MessageTest(msg_id='single-space-above-raises', node=node_newline_above_returns_raises)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(node_newline_above_returns_raises)

    def test_excessive_newline_above_args(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        node_with_two_newline = astroid.extract_node(u'def func(arg): #@\n                    """Returns something.\n\n\n                    Args:\n                        arg: argument. This is  description.\n\n\n                    Returns:\n                        int. Returns something.\n\n\n                    Yields:\n                        yield_something. Description.\n                    """\n                    return True\n                    yield something\n        ')
        single_space_above_args_message = testutils.MessageTest(msg_id='single-space-above-args', node=node_with_two_newline)
        single_space_above_returns_message = testutils.MessageTest(msg_id='single-space-above-returns', node=node_with_two_newline)
        single_space_above_yields_message = testutils.MessageTest(msg_id='single-space-above-yield', node=node_with_two_newline)
        with self.checker_test_object.assertAddsMessages(single_space_above_args_message, single_space_above_returns_message, single_space_above_yields_message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(node_with_two_newline)

    def test_return_in_comment(self) -> None:
        if False:
            i = 10
            return i + 15
        node_with_return_in_comment = astroid.extract_node(u'def func(arg): #@\n                    """Returns something.\n\n                    Args:\n                        arg: argument. Description.\n\n                    Returns:\n                        returns_something. Description.\n                    """\n                    "Returns: something"\n                    return something\n        ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_functiondef(node_with_return_in_comment)

    def test_function_with_no_args(self) -> None:
        if False:
            print('Hello World!')
        node_with_no_args = astroid.extract_node(u'def func():\n                """Do something."""\n\n                a = 1 + 2\n        ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_functiondef(node_with_no_args)

    def test_well_placed_newline(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        node_with_no_error_message = astroid.extract_node(u'def func(arg): #@\n                    """Returns something.\n\n                    Args:\n                        arg: argument. This is description.\n\n                    Returns:\n                        returns_something. This is description.\n\n                    Raises:\n                        raises. Something.\n\n                    Yields:\n                        yield_something. This is description.\n                    """\n                    raise something\n                    yield something\n                    return something\n        ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_functiondef(node_with_no_error_message)

    def test_invalid_parameter_indentation_in_docstring(self) -> None:
        if False:
            while True:
                i = 10
        raises_invalid_indentation_node = astroid.extract_node(u'def func(arg): #@\n                    """This is a docstring.\n\n                        Raises:\n                        NoVariableException. Variable.\n                    """\n                    Something\n        ')
        message = testutils.MessageTest(msg_id='4-space-indentation-in-docstring', node=raises_invalid_indentation_node)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(raises_invalid_indentation_node)
        return_invalid_indentation_node = astroid.extract_node(u'def func(arg): #@\n                    """This is a docstring.\n\n                        Returns:\n                        str. If :true,\n                            individual key=value pairs.\n                    """\n                    Something\n        ')
        message = testutils.MessageTest(msg_id='4-space-indentation-in-docstring', node=return_invalid_indentation_node)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(return_invalid_indentation_node)

    def test_invalid_description_indentation_docstring(self) -> None:
        if False:
            return 10
        invalid_raises_description_indentation_node = astroid.extract_node(u'def func(arg): #@\n                    """This is a docstring.\n\n                        Raises:\n                            AssertionError. If the\n                            schema is not valid.\n                    """\n                    Something\n        ')
        incorrect_indentation_message = testutils.MessageTest(msg_id='8-space-indentation-in-docstring', node=invalid_raises_description_indentation_node)
        malformed_raises_message = testutils.MessageTest(msg_id='malformed-raises-section', node=invalid_raises_description_indentation_node)
        with self.checker_test_object.assertAddsMessages(incorrect_indentation_message, malformed_raises_message, malformed_raises_message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(invalid_raises_description_indentation_node)
        invalid_return_description_indentation_node = astroid.extract_node(u'def func(arg): #@\n                    """This is a docstring.\n\n                        Returns:\n                            str. If :true,\n                                individual key=value pairs.\n                    """\n                    return Something\n        ')
        message = testutils.MessageTest(msg_id='4-space-indentation-in-docstring', node=invalid_return_description_indentation_node)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(invalid_return_description_indentation_node)
        invalid_yield_description_indentation_node = astroid.extract_node(u'def func(arg): #@\n                    """This is a docstring.\n\n                        Yields:\n                            str. If :true,\n                                incorrent indentation line.\n                    """\n                    yield Something\n        ')
        message = testutils.MessageTest(msg_id='4-space-indentation-in-docstring', node=invalid_yield_description_indentation_node)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(invalid_yield_description_indentation_node)

    def test_malformed_parameter_docstring(self) -> None:
        if False:
            i = 10
            return i + 15
        invalid_parameter_name = astroid.extract_node(u'def func(arg): #@\n                    """This is a docstring.\n\n                        Raises:\n                            Incorrect-Exception. If the\n                            schema is not valid.\n                    """\n                    Something\n        ')
        malformed_raises_message = testutils.MessageTest(msg_id='malformed-raises-section', node=invalid_parameter_name)
        with self.checker_test_object.assertAddsMessages(malformed_raises_message, malformed_raises_message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(invalid_parameter_name)

    def test_well_formed_single_line_docstring(self) -> None:
        if False:
            print('Hello World!')
        node_with_no_error_message = astroid.extract_node(u'def func(arg): #@\n                    """This is a docstring."""\n                    Something\n        ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_functiondef(node_with_no_error_message)

    def test_well_formed_multi_line_docstring(self) -> None:
        if False:
            i = 10
            return i + 15
        node_with_no_error_message = astroid.extract_node(u'def func(arg): #@\n                    """This is a docstring.\n\n                        Args:\n                            arg: variable. Description.\n                    """\n                    Something\n        ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_functiondef(node_with_no_error_message)

    def test_well_formed_multi_line_description_docstring(self) -> None:
        if False:
            i = 10
            return i + 15
        node_with_no_error_message = astroid.extract_node(u'def func(arg): #@\n                    """This is a docstring.\n\n                        Args:\n                            arg: bool. If true, individual key=value\n                                pairs separated by \'&\' are\n                                generated for each element of the value\n                                sequence for the key.\n                    """\n                    Something\n        ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_functiondef(node_with_no_error_message)
        node_with_no_error_message = astroid.extract_node(u'def func(arg): #@\n                    """This is a docstring.\n\n                        Raises:\n                            doseq. If true, individual\n                                key=value pairs separated by \'&\' are\n                                generated for each element of\n                                the value sequence for the key\n                                temp temp temp temp.\n                            query. The query to be encoded.\n                    """\n                    Something\n        ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_functiondef(node_with_no_error_message)
        node_with_no_error_message = astroid.extract_node(u'def func(arg):\n                    """This is a docstring.\n\n                        Returns:\n                            str. The string parsed using\n                            Jinja templating. Returns an error\n                            string in case of error in parsing.\n\n                        Yields:\n                            tuple. For ExplorationStatsModel,\n                            a 2-tuple of the form (exp_id, value)\n                            where value is of the form.\n                    """\n                    if True:\n                        return Something\n                    else:\n                        yield something\n        ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_functiondef(node_with_no_error_message)
        node_with_no_error_message = astroid.extract_node(u'def func(arg): #@\n                    """This is a docstring.\n\n                        Returns:\n                            str. From this item there\n                            is things:\n                                Jinja templating. Returns an error\n                            string in case of error in parsing.\n\n                        Yields:\n                            tuple. For ExplorationStatsModel:\n                                {key\n                                    (sym)\n                                }.\n                    """\n                    if True:\n                        return Something\n                    else:\n                        yield (a, b)\n        ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_functiondef(node_with_no_error_message)

    def test_checks_args_formatting_docstring(self) -> None:
        if False:
            i = 10
            return i + 15
        self.checker_test_object = testutils.CheckerTestCase()
        self.checker_test_object.CHECKER_CLASS = pylint_extensions.DocstringParameterChecker
        self.checker_test_object.setup_method()
        invalid_args_description_node = astroid.extract_node('\n        def func(test_var_one, test_var_two): #@\n            """Function to test docstring parameters.\n\n            Args:\n                test_var_one: int. First test variable.\n                test_var_two: str. Second test variable.\n                Incorrect description indentation\n\n            Returns:\n                int. The test result.\n            """\n            result = test_var_one + test_var_two\n            return result\n        ')
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='8-space-indentation-for-arg-in-descriptions-doc', node=invalid_args_description_node, args='Incorrect'), testutils.MessageTest(msg_id='malformed-args-section', node=invalid_args_description_node), ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(invalid_args_description_node)
        invalid_param_indentation_node = astroid.extract_node('\n        def func(test_var_one): #@\n            """Function to test docstring parameters.\n\n            Args:\n                 test_var_one: int. First test variable.\n\n            Returns:\n                int. The test result.\n            """\n            result = test_var_one + test_var_two\n            return result\n        ')
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='4-space-indentation-for-arg-parameters-doc', node=invalid_param_indentation_node, args='test_var_one:'), ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(invalid_param_indentation_node)
        invalid_header_indentation_node = astroid.extract_node('\n        def func(test_var_one): #@\n            """Function to test docstring parameters.\n\n             Args:\n                 test_var_one: int. First test variable.\n\n            Returns:\n                int. The test result.\n            """\n            result = test_var_one + test_var_two\n            return result\n        ')
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='incorrect-indentation-for-arg-header-doc', node=invalid_header_indentation_node), ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(invalid_header_indentation_node)

    def test_correct_args_formatting_docstring(self) -> None:
        if False:
            while True:
                i = 10
        self.checker_test_object = testutils.CheckerTestCase()
        self.checker_test_object.CHECKER_CLASS = pylint_extensions.DocstringParameterChecker
        self.checker_test_object.setup_method()
        valid_free_form_node = astroid.extract_node('\n        def func(test_var_one, test_var_two): #@\n            """Function to test docstring parameters.\n\n            Args:\n                test_var_one: int. First test variable.\n                test_var_two: str. Second test variable:\n                    Incorrect description indentation\n                        {\n                            key:\n                        }.\n\n            Returns:\n                int. The test result.\n            """\n            result = test_var_one + test_var_two\n            return result\n        ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_functiondef(valid_free_form_node)
        valid_indentation_node = astroid.extract_node('\n        def func(test_var_one, test_var_two): #@\n            """Function to test docstring parameters.\n\n            Args:\n                test_var_one: int. First test variable.\n                test_var_two: str. Second test variable:\n                    Correct indentaion.\n\n            Returns:\n                int. The test result.\n            """\n            result = test_var_one + test_var_two\n            return result\n        ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_functiondef(valid_indentation_node)
        valid_indentation_with_kw_args_node = astroid.extract_node('\n        def func( #@\n            test_var_one,\n            *,\n            test_var_two\n        ):\n            """Function to test docstring parameters.\n\n            Args:\n                test_var_one: int. First test variable.\n                test_var_two: str. Second test variable.\n\n            Returns:\n                int. The test result.\n            """\n            result = test_var_one + test_var_two\n            return result\n        ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_functiondef(valid_indentation_with_kw_args_node)

    def test_finds_docstring_parameter(self) -> None:
        if False:
            while True:
                i = 10
        self.checker_test_object = testutils.CheckerTestCase()
        self.checker_test_object.CHECKER_CLASS = pylint_extensions.DocstringParameterChecker
        self.checker_test_object.setup_method()
        (valid_func_node, valid_return_node) = astroid.extract_node('\n        def test(test_var_one, test_var_two): #@\n            """Function to test docstring parameters.\n\n            Args:\n                test_var_one: int. First test variable.\n                test_var_two: str. Second test variable.\n\n            Returns:\n                int. The test result.\n            """\n            result = test_var_one + test_var_two\n            return result #@\n        ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_functiondef(valid_func_node)
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_return(valid_return_node)
        (valid_func_node, valid_yield_node) = astroid.extract_node('\n        def test(test_var_one, test_var_two): #@\n            """Function to test docstring parameters."""\n            result = test_var_one + test_var_two\n            yield result #@\n        ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_functiondef(valid_func_node)
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_yield(valid_yield_node)
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_return(valid_yield_node)
        (missing_yield_type_func_node, missing_yield_type_yield_node) = astroid.extract_node('\n        class Test:\n            def __init__(self, test_var_one, test_var_two): #@\n                """Function to test docstring parameters.\n\n                Args:\n                    test_var_one: int. First test variable.\n                    test_var_two: str. Second test variable.\n\n                Returns:\n                    int. The test result.\n                """\n                result = test_var_one + test_var_two\n                yield result #@\n        ')
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='redundant-returns-doc', node=missing_yield_type_func_node), ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(missing_yield_type_func_node)
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='missing-yield-doc', node=missing_yield_type_func_node), testutils.MessageTest(msg_id='missing-yield-type-doc', node=missing_yield_type_func_node), ignore_position=True):
            self.checker_test_object.checker.visit_yieldfrom(missing_yield_type_yield_node)
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_return(missing_yield_type_yield_node)
        (missing_return_type_func_node, missing_return_type_return_node) = astroid.extract_node('\n        class Test:\n            def __init__(self, test_var_one, test_var_two): #@\n                """Function to test docstring parameters.\n\n                Args:\n                    test_var_one: int. First test variable.\n                    test_var_two: str. Second test variable.\n\n                Yields:\n                    int. The test result.\n                """\n                result = test_var_one + test_var_two\n                return result #@\n        ')
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='redundant-yields-doc', node=missing_return_type_func_node), ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(missing_return_type_func_node)
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='missing-return-doc', node=missing_return_type_func_node), testutils.MessageTest(msg_id='missing-return-type-doc', node=missing_return_type_func_node), ignore_position=True):
            self.checker_test_object.checker.visit_return(missing_return_type_return_node)
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_yield(missing_return_type_return_node)
        valid_raise_node = astroid.extract_node('\n        def func(test_var_one, test_var_two):\n            """Function to test docstring parameters.\n\n            Args:\n                test_var_one: int. First test variable.\n                test_var_two: str. Second test variable.\n\n            Raises:\n                Exception. An exception.\n            """\n            raise Exception #@\n        ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_raise(valid_raise_node)
        (missing_raise_type_func_node, missing_raise_type_raise_node) = astroid.extract_node('\n        def func(test_var_one, test_var_two): #@\n            """Function to test raising exceptions.\n\n            Args:\n                test_var_one: int. First test variable.\n                test_var_two: str. Second test variable.\n            """\n            raise Exception #@\n        ')
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='missing-raises-doc', args=('Exception',), node=missing_raise_type_func_node), ignore_position=True):
            self.checker_test_object.checker.visit_raise(missing_raise_type_raise_node)
        valid_raise_node = astroid.extract_node('\n        class Test:\n            raise Exception #@\n        ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_raise(valid_raise_node)
        valid_raise_node = astroid.extract_node('\n        class Test():\n            @property\n            def decorator_func(self):\n                pass\n\n            @decorator_func.setter\n            @property\n            def func(self):\n                raise Exception #@\n        ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_raise(valid_raise_node)
        valid_raise_node = astroid.extract_node('\n        class Test():\n            def func(self):\n                raise Exception #@\n        ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_raise(valid_raise_node)
        valid_raise_node = astroid.extract_node('\n        def func():\n            try:\n                raise Exception #@\n            except Exception:\n                pass\n        ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_raise(valid_raise_node)
        valid_raise_node = astroid.extract_node('\n        def func():\n            """Function to test raising exceptions."""\n            raise Exception #@\n        ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_raise(valid_raise_node)
        valid_raise_node = astroid.extract_node('\n        def my_func(self):\n            """This is a docstring.\n            :raises NameError: Never.\n            """\n            def ex_func(val):\n                return RuntimeError(val)\n            raise ex_func(\'hi\') #@\n            raise NameError(\'hi\')\n        ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_raise(valid_raise_node)
        valid_raise_node = astroid.extract_node('\n        from unknown import Unknown\n        def my_func(self):\n            """This is a docstring.\n            :raises NameError: Never.\n            """\n            raise Unknown(\'hi\') #@\n            raise NameError(\'hi\')\n        ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_raise(valid_raise_node)
        valid_raise_node = astroid.extract_node('\n        def my_func(self):\n            """This is a docstring.\n            :raises NameError: Never.\n            """\n            def ex_func(val):\n                def inner_func(value):\n                    return OSError(value)\n                return RuntimeError(val)\n            raise ex_func(\'hi\') #@\n            raise NameError(\'hi\')\n        ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_raise(valid_raise_node)
        valid_return_node = astroid.extract_node('\n        def func():\n            """Function to test return values."""\n            return None #@\n        ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_return(valid_return_node)
        valid_return_node = astroid.extract_node('\n        def func():\n            """Function to test return values."""\n            return #@\n        ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_return(valid_return_node)
        missing_param_func_node = astroid.extract_node('\n        def func(test_var_one, test_var_two, *args, **kwargs): #@\n            """Function to test docstring parameters.\n\n            Args:\n                test_var_one: int. First test variable.\n                test_var_two: str. Second test variable.\n\n            Returns:\n                int. The test result.\n            """\n            result = test_var_one + test_var_two\n            return result\n        ')
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='missing-param-doc', node=missing_param_func_node, args=('args, kwargs',)), ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(missing_param_func_node)
        missing_param_func_node = astroid.extract_node('\n        def func(test_var_one, test_var_two): #@\n            """Function to test docstring parameters.\n\n            Args:\n                test_var_one: int. First test variable.\n                invalid_var_name: str. Second test variable.\n\n            Returns:\n                int. The test result.\n            """\n            result = test_var_one + test_var_two\n            return result\n        ')
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='missing-param-doc', node=missing_param_func_node, args=('test_var_two',)), testutils.MessageTest(msg_id='missing-type-doc', node=missing_param_func_node, args=('test_var_two',)), testutils.MessageTest(msg_id='differing-param-doc', node=missing_param_func_node, args=('invalid_var_name',)), testutils.MessageTest(msg_id='differing-type-doc', node=missing_param_func_node, args=('invalid_var_name',)), testutils.MessageTest(msg_id='8-space-indentation-for-arg-in-descriptions-doc', node=missing_param_func_node, args='invalid_var_name:'), ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(missing_param_func_node)
        (class_node, multiple_constructor_func_node) = astroid.extract_node('\n        class Test(): #@\n            """Function to test docstring parameters.\n\n            Args:\n                test_var_one: int. First test variable.\n                test_var_two: str. Second test variable.\n\n            Returns:\n                int. The test result.\n            """\n\n            def __init__(self, test_var_one, test_var_two): #@\n                """Function to test docstring parameters.\n\n                Args:\n                    test_var_one: int. First test variable.\n                    test_var_two: str. Second test variable.\n\n                Returns:\n                    int. The test result.\n                """\n                result = test_var_one + test_var_two\n                return result\n        ')
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='multiple-constructor-doc', node=class_node, args=(class_node.name,)), ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(multiple_constructor_func_node)

    def test_visit_raise_warns_unknown_style(self) -> None:
        if False:
            return 10
        self.checker_test_object.checker.config.accept_no_raise_doc = False
        node = astroid.extract_node('\n        def my_func(self):\n            """This is a docstring."""\n            raise RuntimeError(\'hi\')\n        ')
        raise_node = node.body[0]
        func_node = raise_node.frame()
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='missing-raises-doc', args=('RuntimeError',), node=func_node), ignore_position=True):
            self.checker_test_object.checker.visit_raise(raise_node)

class ImportOnlyModulesCheckerTests(unittest.TestCase):

    def test_finds_import_from(self) -> None:
        if False:
            while True:
                i = 10
        checker_test_object = testutils.CheckerTestCase()
        checker_test_object.CHECKER_CLASS = pylint_extensions.ImportOnlyModulesChecker
        checker_test_object.setup_method()
        importfrom_node1 = astroid.extract_node('\n            from os import path #@\n            import sys\n        ')
        with checker_test_object.assertNoMessages():
            checker_test_object.checker.visit_importfrom(importfrom_node1)
        importfrom_node2 = astroid.extract_node('\n            from os import error #@\n            import sys\n        ')
        with checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='import-only-modules', node=importfrom_node2, args=('error', 'os')), ignore_position=True):
            checker_test_object.checker.visit_importfrom(importfrom_node2)
        importfrom_node3 = astroid.extract_node('\n            from invalid_module import invalid_module #@\n        ')
        with checker_test_object.assertNoMessages():
            checker_test_object.checker.visit_importfrom(importfrom_node3)
        importfrom_node4 = astroid.extract_node('\n            from constants import constants #@\n        ')
        with checker_test_object.assertNoMessages():
            checker_test_object.checker.visit_importfrom(importfrom_node4)
        importfrom_node5 = astroid.extract_node('\n            from os import invalid_module #@\n        ')
        with checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='import-only-modules', node=importfrom_node5, args=('invalid_module', 'os')), ignore_position=True):
            checker_test_object.checker.visit_importfrom(importfrom_node5)
        importfrom_node6 = astroid.extract_node('\n            from .constants import constants #@\n        ', module_name='.constants')
        with checker_test_object.assertNoMessages():
            checker_test_object.checker.visit_importfrom(importfrom_node6)

    def test_importing_internals_from_allowed_modules_does_not_raise_message(self) -> None:
        if False:
            while True:
                i = 10
        checker_test_object = testutils.CheckerTestCase()
        checker_test_object.CHECKER_CLASS = pylint_extensions.ImportOnlyModulesChecker
        checker_test_object.setup_method()
        importfrom_node = astroid.extract_node('\n            from __future__ import invalid_module #@\n        ')
        with checker_test_object.assertNoMessages():
            checker_test_object.checker.visit_importfrom(importfrom_node)

class BackslashContinuationCheckerTests(unittest.TestCase):

    def test_finds_backslash_continuation(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        checker_test_object = testutils.CheckerTestCase()
        checker_test_object.CHECKER_CLASS = pylint_extensions.BackslashContinuationChecker
        checker_test_object.setup_method()
        node = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u"message1 = 'abc'\\\n'cde'\\\n'xyz'\n                message2 = 'abc\\\\'\n                message3 = (\n                    'abc\\\\'\n                    'xyz\\\\'\n                )\n                ")
        node.file = filename
        node.path = filename
        checker_test_object.checker.process_module(node)
        with checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='backslash-continuation', line=1), testutils.MessageTest(msg_id='backslash-continuation', line=2)):
            temp_file.close()

class FunctionArgsOrderCheckerTests(unittest.TestCase):

    def test_finds_function_def(self) -> None:
        if False:
            while True:
                i = 10
        checker_test_object = testutils.CheckerTestCase()
        checker_test_object.CHECKER_CLASS = pylint_extensions.FunctionArgsOrderChecker
        checker_test_object.setup_method()
        functiondef_node1 = astroid.extract_node('\n        def test(self,test_var_one, test_var_two): #@\n            result = test_var_one + test_var_two\n            return result\n        ')
        with checker_test_object.assertNoMessages():
            checker_test_object.checker.visit_functiondef(functiondef_node1)
        functiondef_node2 = astroid.extract_node('\n        def test(test_var_one, test_var_two, self): #@\n            result = test_var_one + test_var_two\n            return result\n        ')
        with checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='function-args-order-self', node=functiondef_node2), ignore_position=True):
            checker_test_object.checker.visit_functiondef(functiondef_node2)
        functiondef_node3 = astroid.extract_node('\n        def test(test_var_one, test_var_two, cls): #@\n            result = test_var_one + test_var_two\n            return result\n        ')
        with checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='function-args-order-cls', node=functiondef_node3), ignore_position=True):
            checker_test_object.checker.visit_functiondef(functiondef_node3)

class RestrictedImportCheckerTests(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.checker_test_object = testutils.CheckerTestCase()
        self.checker_test_object.CHECKER_CLASS = pylint_extensions.RestrictedImportChecker
        self.checker_test_object.setup_method()
        self.checker_test_object.checker.config.forbidden_imports = ('*core.controllers*:\n    import core.platform*   |  \n    import core.storage*\n', '*core.domain*:import core.controllers*', '   *core.storage*:import    core.domain*   ', '*core.domain.*_domain:\n    from core.domain    import    *_service*   |\n    from   core.domain import *_cleaner|\n      from core.domain import *_registry |\n    from core.domain import *_fetchers  |\n    from core.domain import *_manager |\n       from core.platform import   models')
        self.checker_test_object.checker.open()

    def test_forbid_domain_import_in_storage_module(self) -> None:
        if False:
            print('Hello World!')
        node_err_import = astroid.extract_node('\n            import core.domain.activity_domain #@\n            ')
        node_err_import.root().name = 'oppia.core.storage.topic'
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='invalid-import', node=node_err_import, args=('core.domain*', '*core.storage*')), ignore_position=True):
            self.checker_test_object.checker.visit_import(node_err_import)

    def test_allow_platform_import_in_storage_module(self) -> None:
        if False:
            print('Hello World!')
        node_no_err_import = astroid.extract_node('\n            import core.platform.email.mailgun_email_services #@\n        ')
        node_no_err_import.root().name = 'oppia.core.storage.topic'
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_import(node_no_err_import)

    def test_forbid_domain_from_import_in_storage_module(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        node_err_importfrom = astroid.extract_node('\n            from core.domain import activity_domain #@\n        ')
        node_err_importfrom.root().name = 'oppia.core.storage.topic'
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='invalid-import', node=node_err_importfrom, args=('core.domain*', '*core.storage*')), ignore_position=True):
            self.checker_test_object.checker.visit_importfrom(node_err_importfrom)

    def test_allow_platform_from_import_in_storage_module(self) -> None:
        if False:
            return 10
        node_no_err_importfrom = astroid.extract_node('\n            from core.platform.email import mailgun_email_services #@\n        ')
        node_no_err_importfrom.root().name = 'oppia.core.storage.topicl'
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_importfrom(node_no_err_importfrom)

    def test_forbid_controllers_import_in_domain_module(self) -> None:
        if False:
            return 10
        node_err_import = astroid.extract_node('\n            import core.controllers.acl_decorators #@\n        ')
        node_err_import.root().name = 'oppia.core.domain'
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='invalid-import', node=node_err_import, args=('core.controllers*', '*core.domain*')), ignore_position=True):
            self.checker_test_object.checker.visit_import(node_err_import)

    def test_allow_platform_import_in_domain_module(self) -> None:
        if False:
            return 10
        node_no_err_import = astroid.extract_node('\n            import core.platform.email.mailgun_email_services_test #@\n        ')
        node_no_err_import.root().name = 'oppia.core.domain'
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_import(node_no_err_import)

    def test_forbid_controllers_from_import_in_domain_module(self) -> None:
        if False:
            while True:
                i = 10
        node_err_importfrom = astroid.extract_node('\n            from core.controllers import acl_decorators #@\n            ')
        node_err_importfrom.root().name = 'oppia.core.domain'
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='invalid-import', node=node_err_importfrom, args=('core.controllers*', '*core.domain*')), ignore_position=True):
            self.checker_test_object.checker.visit_importfrom(node_err_importfrom)

    def test_allow_platform_from_import_in_domain_module(self) -> None:
        if False:
            print('Hello World!')
        node_no_err_importfrom = astroid.extract_node('\n            from core.platform.email import mailgun_email_services_test #@\n        ')
        node_no_err_importfrom.root().name = 'oppia.core.domain'
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_importfrom(node_no_err_importfrom)

    def test_forbid_service_import_in_domain_file(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        node_err_import = astroid.extract_node('\n            import core.domain.exp_services #@\n            ')
        node_err_import.root().name = 'oppia.core.domain.exp_domain'
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='invalid-import-from', node=node_err_import, args=('*_service*', 'core.domain', '*core.domain.*_domain')), ignore_position=True):
            self.checker_test_object.checker.visit_import(node_err_import)

    def test_allow_domain_file_import_in_domain_file(self) -> None:
        if False:
            while True:
                i = 10
        node_no_err_import = astroid.extract_node('\n            import core.domain.collection_domain #@\n            ')
        node_no_err_import.root().name = 'oppia.core.domain.topic_domain'
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_import(node_no_err_import)

    def test_forbid_cleaner_from_import_in_domain_file(self) -> None:
        if False:
            while True:
                i = 10
        node_err_importfrom = astroid.extract_node('\n            from core.domain import html_cleaner #@\n            ')
        node_err_importfrom.root().name = 'oppia.core.domain.collection_domain'
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='invalid-import-from', node=node_err_importfrom, args=('*_cleaner', 'core.domain', '*core.domain.*_domain')), ignore_position=True):
            self.checker_test_object.checker.visit_importfrom(node_err_importfrom)

    def test_allow_domain_file_from_import_in_domain_file(self) -> None:
        if False:
            print('Hello World!')
        node_no_err_importfrom = astroid.extract_node('\n            from core.domain import exp_domain #@\n            ')
        node_no_err_importfrom.root().name = 'oppia.core.domain.story_domain'
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_importfrom(node_no_err_importfrom)

    def test_forbid_platform_import_in_controllers_module(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        node_err_import = astroid.extract_node('\n            import core.platform #@\n        ')
        node_err_import.root().name = 'oppia.core.controllers.controller'
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='invalid-import', node=node_err_import, args=('core.platform*', '*core.controllers*')), ignore_position=True):
            self.checker_test_object.checker.visit_import(node_err_import)

    def test_forbid_storage_import_in_controllers_module(self) -> None:
        if False:
            print('Hello World!')
        node_err_import = astroid.extract_node('\n            import core.storage #@\n        ')
        node_err_import.root().name = 'oppia.core.controllers.controller'
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='invalid-import', node=node_err_import, args=('core.storage*', '*core.controllers*')), ignore_position=True):
            self.checker_test_object.checker.visit_import(node_err_import)

    def test_allow_domain_import_in_controllers_module(self) -> None:
        if False:
            print('Hello World!')
        node_no_err_import = astroid.extract_node('\n            import core.domain #@\n        ')
        node_no_err_import.root().name = 'oppia.core.controllers.controller'
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_import(node_no_err_import)

    def test_forbid_platform_from_import_in_controllers_module(self) -> None:
        if False:
            while True:
                i = 10
        node_no_err_importfrom = astroid.extract_node('\n            from core.platform import models #@\n        ')
        node_no_err_importfrom.root().name = 'oppia.core.controllers.controller'
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='invalid-import', node=node_no_err_importfrom, args=('core.platform*', '*core.controllers*')), ignore_position=True):
            self.checker_test_object.checker.visit_importfrom(node_no_err_importfrom)

    def test_forbid_storage_from_import_in_controllers_module(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        node_no_err_importfrom = astroid.extract_node('\n            from core.storage.user import gae_models as user_models #@\n        ')
        node_no_err_importfrom.root().name = 'oppia.core.controllers.controller'
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='invalid-import', node=node_no_err_importfrom, args=('core.storage*', '*core.controllers*')), ignore_position=True):
            self.checker_test_object.checker.visit_importfrom(node_no_err_importfrom)

    def test_allow_domain_from_import_in_controllers_module(self) -> None:
        if False:
            i = 10
            return i + 15
        node_no_err_importfrom = astroid.extract_node('\n            from core.domain import user_services #@\n        ')
        node_no_err_importfrom.root().name = 'oppia.core.controllers.controller'
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_importfrom(node_no_err_importfrom)

class SingleCharAndNewlineAtEOFCheckerTests(unittest.TestCase):

    def test_checks_single_char_and_newline_eof(self) -> None:
        if False:
            i = 10
            return i + 15
        checker_test_object = testutils.CheckerTestCase()
        checker_test_object.CHECKER_CLASS = pylint_extensions.SingleCharAndNewlineAtEOFChecker
        checker_test_object.setup_method()
        node_missing_newline_at_eof = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u"c = 'something dummy'\n                ")
        node_missing_newline_at_eof.file = filename
        node_missing_newline_at_eof.path = filename
        checker_test_object.checker.process_module(node_missing_newline_at_eof)
        with checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='newline-at-eof', line=2)):
            temp_file.close()
        node_single_char_file = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'1')
        node_single_char_file.file = filename
        node_single_char_file.path = filename
        checker_test_object.checker.process_module(node_single_char_file)
        with checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='only-one-character', line=1)):
            temp_file.close()
        node_with_no_error_message = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u"x = 'something dummy'")
        node_with_no_error_message.file = filename
        node_with_no_error_message.path = filename
        checker_test_object.checker.process_module(node_with_no_error_message)
        with checker_test_object.assertNoMessages():
            temp_file.close()

class TypeIgnoreCommentCheckerTests(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        self.checker_test_object = testutils.CheckerTestCase()
        self.checker_test_object.CHECKER_CLASS = pylint_extensions.TypeIgnoreCommentChecker
        self.checker_test_object.setup_method()
        self.checker_test_object.checker.config.allowed_type_ignore_error_codes = ['attr-defined', 'union-attr', 'arg-type', 'call-overload', 'override', 'return', 'assignment', 'list-item', 'dict-item', 'typeddict-item', 'func-returns-value', 'misc', 'type-arg', 'no-untyped-def', 'no-untyped-call', 'no-any-return']

    def test_type_ignore_used_without_comment_raises_error(self) -> None:
        if False:
            print('Hello World!')
        node_function_with_type_ignore_only = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                suggestion.change.new_value = (  # type: ignore[attr-defined]\n                    new_content\n                ) #@\n                ')
        node_function_with_type_ignore_only.file = filename
        message = testutils.MessageTest(msg_id='mypy-ignore-used', line=2, node=node_function_with_type_ignore_only)
        with self.checker_test_object.assertAddsMessages(message):
            self.checker_test_object.checker.visit_module(node_function_with_type_ignore_only)
        temp_file.close()

    def test_raises_error_if_prohibited_error_code_is_used(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        node_with_prohibited_error_code = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                suggestion.change.new_value = (  # type: ignore[some-new-ignore]\n                    new_content\n                ) #@\n                ')
        node_with_prohibited_error_code.file = filename
        message = testutils.MessageTest(msg_id='prohibited-type-ignore-used', line=2, node=node_with_prohibited_error_code, args=('some-new-ignore',))
        with self.checker_test_object.assertAddsMessages(message):
            self.checker_test_object.checker.visit_module(node_with_prohibited_error_code)
        temp_file.close()
        node_with_prohibited_type_ignore_error_code = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u"\n                # Here we use MyPy ignore because ...\n                suggestion.change.new_value = (  # type: ignore[attr-defined]\n                    new_content\n                )\n\n                suggestion.change.new_value = (  # type: ignore[truthy-bool]\n                    new_content\n                )\n\n                # Here we use MyPy ignore because ...\n                func_only_accept_str('hi')  # type: ignore[attr-defined]\n\n                #@\n                ")
        node_with_prohibited_type_ignore_error_code.file = filename
        message = testutils.MessageTest(msg_id='prohibited-type-ignore-used', line=7, node=node_with_prohibited_type_ignore_error_code, args=('truthy-bool',))
        with self.checker_test_object.assertAddsMessages(message):
            self.checker_test_object.checker.visit_module(node_with_prohibited_type_ignore_error_code)
        temp_file.close()

    def test_raises_error_if_prohibited_error_code_is_used_in_combined_form(self) -> None:
        if False:
            print('Hello World!')
        node_with_prohibited_error_code_in_combined_form = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                suggestion.change.new_value = (  # type: ignore[arg-type, no-untyped-call, truthy-bool] pylint: disable=line-too-long\n                    new_content\n                ) #@\n                ')
        node_with_prohibited_error_code_in_combined_form.file = filename
        message = testutils.MessageTest(msg_id='prohibited-type-ignore-used', line=2, node=node_with_prohibited_error_code_in_combined_form, args=('truthy-bool',))
        with self.checker_test_object.assertAddsMessages(message):
            self.checker_test_object.checker.visit_module(node_with_prohibited_error_code_in_combined_form)
        temp_file.close()
        node_with_multiple_prohibited_error_code_in_combined_form = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                suggestion.change.new_value = (  # type: ignore[return-none, no-untyped-call, truthy-bool] pylint: disable=line-too-long\n                    new_content\n                ) #@\n                ')
        node_with_multiple_prohibited_error_code_in_combined_form.file = filename
        message = testutils.MessageTest(msg_id='prohibited-type-ignore-used', line=2, node=node_with_multiple_prohibited_error_code_in_combined_form, args=('return-none', 'truthy-bool'))
        with self.checker_test_object.assertAddsMessages(message):
            self.checker_test_object.checker.visit_module(node_with_multiple_prohibited_error_code_in_combined_form)
        temp_file.close()

    def test_extra_type_ignore_comment_used_in_a_module_raises_error(self) -> None:
        if False:
            return 10
        node_function_with_extra_comment = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u"\n                # Here we use MyPy ignore because ...\n                suggestion.change.new_value = (   # type: ignore[attr-defined]\n                    new_content\n                )\n\n                # Here we use MyPy ignore because ...\n                suggestion.change.new_value = (\n                    new_content\n                )\n\n                # Here we use MyPy ignore because ...\n                func_only_accept_str('hi')   # type: ignore[attr-defined]\n\n                # Here we use MyPy ignore because ...\n                suggestion.change.new_value = (\n                    new_content\n                )\n                #@\n                ")
        node_function_with_extra_comment.file = filename
        message1 = testutils.MessageTest(msg_id='redundant-type-comment', line=7, node=node_function_with_extra_comment)
        message2 = testutils.MessageTest(msg_id='redundant-type-comment', line=15, node=node_function_with_extra_comment)
        with self.checker_test_object.assertAddsMessages(message1, message2):
            self.checker_test_object.checker.visit_module(node_function_with_extra_comment)
        temp_file.close()
        node_function_with_extra_comment2 = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u"\n                # Here we use MyPy ignore because ...\n                suggestion.change.new_value = (   # type: ignore[attr-defined]\n                    new_content\n                )\n\n                # Here we use MyPy ignore because ...\n                suggestion.change.new_value = (\n                    new_content\n                )\n\n                # Here we use MyPy ignore because ...\n                func_only_accept_str('hi')   # type: ignore[attr-defined]\n                #@\n                ")
        node_function_with_extra_comment2.file = filename
        message = testutils.MessageTest(msg_id='redundant-type-comment', line=7, node=node_function_with_extra_comment2)
        with self.checker_test_object.assertAddsMessages(message):
            self.checker_test_object.checker.visit_module(node_function_with_extra_comment2)
        temp_file.close()

    def test_raises_error_if_type_ignore_is_in_second_place(self) -> None:
        if False:
            return 10
        node_with_type_ignore = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                suggestion.change.new_value = (  # pylint: disable=line-too-long type: ignore[attr-defined]\n                    new_content\n                )\n                #@\n                ')
        node_with_type_ignore.file = filename
        message = testutils.MessageTest(msg_id='mypy-ignore-used', line=2, node=node_with_type_ignore)
        with self.checker_test_object.assertAddsMessages(message):
            self.checker_test_object.checker.visit_module(node_with_type_ignore)
        temp_file.close()

    def test_type_ignores_with_comments_should_not_raises_error(self) -> None:
        if False:
            i = 10
            return i + 15
        node_with_type_ignore_in_single_form = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                # Here we use MyPy ignore because attributes on BaseChange\n                # class are defined dynamically.\n                suggestion.change.new_value = (  # type: ignore[attr-defined]\n                    new_content\n                )\n\n                # Here we use MyPy ignore because this function is can only\n                # str values but here we are providing integer which causes\n                # MyPy to throw an error. Thus to avoid the error, we used\n                # ignore here.\n                func_only_accept_str(1234)  # type: ignore[arg-type] #@\n                ')
        node_with_type_ignore_in_single_form.file = filename
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_module(node_with_type_ignore_in_single_form)
        temp_file.close()
        node_with_type_ignore_in_combined_form = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                # Here we use MyPy ignore because ...\n                suggestion.change.new_value = (  # type: ignore[attr-defined, list-item]\n                    new_content\n                )\n\n                # Here we use MyPy ignore because ...\n                func_only_accept_str(1234)  # type: ignore[arg-type]\n                #@\n                ')
        node_with_type_ignore_in_combined_form.file = filename
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_module(node_with_type_ignore_in_combined_form)
        temp_file.close()

    def test_untyped_call_type_ignores_should_not_raise_error(self) -> None:
        if False:
            return 10
        node_function = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                # Here we use MyPy ignore because attributes on BaseChange\n                # class are defined dynamically.\n                suggestion.change.new_value = (  # type: ignore[attr-defined]\n                    new_content\n                )\n\n                func_only_accept_str(1234)  # type: ignore[no-untyped-call] #@\n                ')
        node_function.file = filename
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_module(node_function)
        temp_file.close()

    def test_raises_error_if_gap_in_ignore_and_comment_is_more_than_fifteen(self) -> None:
        if False:
            i = 10
            return i + 15
        node_with_ignore_and_more_than_fifteen_gap = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u"\n                # Here we use MyPy ignore because stubs of protobuf are not\n                # available yet.\n\n                variable_one: str = '123'\n                variable_two: str = '1234'\n                # Some other content of module one.\n\n                # Line 1 content.\n                # Line 2 content.\n                # Line 3 content.\n                # Line 4 content.\n\n                # Some other content of module two.\n\n                def test_foo(arg: str) -> str:\n\n                def foo(exp_id: str) -> str:  # type: ignore[arg-type]\n                    return 'hi' #@\n                ")
        node_with_ignore_and_more_than_fifteen_gap.file = filename
        message1 = testutils.MessageTest(msg_id='mypy-ignore-used', line=18, node=node_with_ignore_and_more_than_fifteen_gap)
        message2 = testutils.MessageTest(msg_id='redundant-type-comment', line=2, node=node_with_ignore_and_more_than_fifteen_gap)
        with self.checker_test_object.assertAddsMessages(message1, message2):
            self.checker_test_object.checker.visit_module(node_with_ignore_and_more_than_fifteen_gap)
        temp_file.close()

    def test_generic_type_ignore_raises_pylint_error(self) -> None:
        if False:
            i = 10
            return i + 15
        node_with_generic_type_ignore = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u"\n                # TODO(#sll): Here we use MyPy ignore because stubs of protobuf\n                # are not available yet.\n\n                def foo(exp_id: str) -> str:  # type: ignore\n                    return 'hi' #@\n                ")
        node_with_generic_type_ignore.file = filename
        message1 = testutils.MessageTest(msg_id='generic-mypy-ignore-used', line=5, node=node_with_generic_type_ignore)
        message2 = testutils.MessageTest(msg_id='redundant-type-comment', line=2, node=node_with_generic_type_ignore)
        with self.checker_test_object.assertAddsMessages(message1, message2):
            self.checker_test_object.checker.visit_module(node_with_generic_type_ignore)
        temp_file.close()
        node_with_both_generic_and_non_generic_type_ignores = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u"\n                # TODO(#sll): Here we use MyPy ignore because stubs of protobuf\n                # are not available yet.\n                def foo(exp_id: str) -> str:  # type: ignore[arg-type]\n                    return 'hi' #@\n\n                def foo(exp_id: str) -> str:  # type: ignore\n                    return 'hi' #@\n\n                # TODO(#sll): Here we use MyPy ignore because stubs of protobuf\n                # are not available yet.\n                def foo(exp_id: str) -> str:  # type: ignore[misc]\n                    return 'hi' #@\n                ")
        node_with_both_generic_and_non_generic_type_ignores.file = filename
        message1 = testutils.MessageTest(msg_id='generic-mypy-ignore-used', line=7, node=node_with_both_generic_and_non_generic_type_ignores)
        with self.checker_test_object.assertAddsMessages(message1):
            self.checker_test_object.checker.visit_module(node_with_both_generic_and_non_generic_type_ignores)
        temp_file.close()

    def test_raises_no_error_if_todo_is_present_initially(self) -> None:
        if False:
            i = 10
            return i + 15
        node_with_ignore_having_todo = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u"\n                # TODO(#sll): Here we use MyPy ignore because stubs of protobuf\n                # are not available yet.\n\n                def foo(exp_id: str) -> str:  # type: ignore[arg-type]\n                    return 'hi' #@\n                ")
        node_with_ignore_having_todo.file = filename
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_module(node_with_ignore_having_todo)
        temp_file.close()

class ExceptionalTypesCommentCheckerTests(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.checker_test_object = testutils.CheckerTestCase()
        self.checker_test_object.CHECKER_CLASS = pylint_extensions.ExceptionalTypesCommentChecker
        self.checker_test_object.setup_method()

    def test_raises_error_if_exceptional_types_are_used_without_comment(self) -> None:
        if False:
            while True:
                i = 10
        node_with_any_type = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u"\n                schema_dict: Dict[str, Any] = {\n                    'key': 'value'\n                } #@\n                ")
        node_with_any_type.file = filename
        message = testutils.MessageTest(msg_id='any-type-used', line=2, node=node_with_any_type)
        with self.checker_test_object.assertAddsMessages(message):
            self.checker_test_object.checker.visit_module(node_with_any_type)
        temp_file.close()
        node_with_object_type = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                func(proto_buff_stuff: object) #@\n                ')
        node_with_object_type.file = filename
        message = testutils.MessageTest(msg_id='object-class-used', line=2, node=node_with_object_type)
        with self.checker_test_object.assertAddsMessages(message):
            self.checker_test_object.checker.visit_module(node_with_object_type)
        temp_file.close()
        node_with_cast_method = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                func(cast(str, change.new_value)) #@\n                ')
        node_with_cast_method.file = filename
        message = testutils.MessageTest(msg_id='cast-func-used', line=2, node=node_with_cast_method)
        with self.checker_test_object.assertAddsMessages(message):
            self.checker_test_object.checker.visit_module(node_with_cast_method)
        temp_file.close()

    def test_raises_error_if_exceptional_types_are_combined_in_module(self) -> None:
        if False:
            while True:
                i = 10
        node_with_combined_types = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u"\n                schema_dict: Dict[str, Any] = {\n                    'key': 'value'\n                }\n\n                def func(proto_buff_stuff: object) -> None:\n                    pass\n\n                # Some other contents of the module.\n\n                # Here we use object because to test the linters.\n                new_object: object = 'strong hi'\n\n                # We are not considering this case.\n                var = object()\n                new_string = 'hi'\n\n                change_value = cast(str, change.new_value) #@\n                ")
        node_with_combined_types.file = filename
        message1 = testutils.MessageTest(msg_id='any-type-used', line=2, node=node_with_combined_types)
        message2 = testutils.MessageTest(msg_id='object-class-used', line=6, node=node_with_combined_types)
        message3 = testutils.MessageTest(msg_id='cast-func-used', line=18, node=node_with_combined_types)
        with self.checker_test_object.assertAddsMessages(message1, message3, message2):
            self.checker_test_object.checker.visit_module(node_with_combined_types)
        temp_file.close()

    def test_raises_error_if_any_type_used_in_function_signature(self) -> None:
        if False:
            return 10
        node_with_any_type_arg = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                def foo(*args: Any) -> None:\n                    pass #@\n                ')
        node_with_any_type_arg.file = filename
        message = testutils.MessageTest(msg_id='any-type-used', line=2, node=node_with_any_type_arg)
        with self.checker_test_object.assertAddsMessages(message):
            self.checker_test_object.checker.visit_module(node_with_any_type_arg)
        temp_file.close()
        node_with_any_type_return = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                def foo(*args: str) -> Any:\n                    pass #@\n                ')
        node_with_any_type_return.file = filename
        message = testutils.MessageTest(msg_id='any-type-used', line=2, node=node_with_any_type_return)
        with self.checker_test_object.assertAddsMessages(message):
            self.checker_test_object.checker.visit_module(node_with_any_type_return)
        temp_file.close()
        node_with_any_type_return_and_args = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                def foo(*args: Any) -> Any:\n                    pass #@\n                ')
        node_with_any_type_return_and_args.file = filename
        message = testutils.MessageTest(msg_id='any-type-used', line=2, node=node_with_any_type_return_and_args)
        with self.checker_test_object.assertAddsMessages(message):
            self.checker_test_object.checker.visit_module(node_with_any_type_return_and_args)
        temp_file.close()
        node_with_multiple_any_type_functions = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                def foo(*args: Any) -> Any:\n                    pass\n\n                def foo1(arg1: str) -> int:\n                    pass\n\n                def foo2(*args: str) -> Any:\n                    pass #@\n                ')
        node_with_multiple_any_type_functions.file = filename
        message = testutils.MessageTest(msg_id='any-type-used', line=2, node=node_with_multiple_any_type_functions)
        message2 = testutils.MessageTest(msg_id='any-type-used', line=8, node=node_with_multiple_any_type_functions)
        with self.checker_test_object.assertAddsMessages(message, message2):
            self.checker_test_object.checker.visit_module(node_with_multiple_any_type_functions)
        temp_file.close()

    def test_any_and_cast_will_not_raise_error_in_import(self) -> None:
        if False:
            print('Hello World!')
        node_with_any_and_cast_imported = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                from typing import Any, cast #@\n                ')
        node_with_any_and_cast_imported.file = filename
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_module(node_with_any_and_cast_imported)
        temp_file.close()
        node_with_any_and_cast_in_multi_line_import = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                from typing import (\n                    Any, Dict, List, Optional, cast\n                ) #@\n                ')
        node_with_any_and_cast_in_multi_line_import.file = filename
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_module(node_with_any_and_cast_in_multi_line_import)
        temp_file.close()

    def test_exceptional_types_with_comments_should_not_raise_error(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        node_with_any_type_and_comment = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u"\n                from typing import Any\n\n                # Here we use type Any because, this function can take\n                # any argument.\n                def foo(arg1: Any) -> None\n                    pass\n\n                # Some other contents of the Module.\n                new_var: str = 'hi'\n\n                # Here we use type Any because, schema dicts can accept\n                # any value.\n                schema_dict: Dict[str, Any] = {\n                    'key': 'value'\n                }\n\n                def foo1(arg2: str) -> None\n                    # Here we use type Any because, new_value can accept any\n                    # value.\n                    new_value: Any = 'hi' #@\n                ")
        node_with_any_type_and_comment.file = filename
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_module(node_with_any_type_and_comment)
        temp_file.close()
        node_with_cast_method_and_comment = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u"\n                from typing import Any, cast\n\n                # Here we use type Any because, this function can take\n                # any argument.\n                def foo(arg1: Any) -> None\n                    pass\n\n                # Here we use cast because we are narrowing down the object\n                # to string object.\n                new_var: str = cast(str, object())\n\n                # Here we use type Any because, schema dicts can accept\n                # any value.\n                schema_dict: Dict[str, Any] = {\n                    'key': 'value'\n                }\n\n                # Here we use object because stubs of protobuf are not\n                # available yet. So, instead of Any we used object here.\n                def save_classifier_data(\n                    exp_id: str,\n                    job_id: str,\n                    classifier_data_proto: object\n                ) -> None:\n                    pass #@\n                ")
        node_with_cast_method_and_comment.file = filename
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_module(node_with_cast_method_and_comment)
        temp_file.close()

    def test_no_error_raised_if_objects_are_present_with_comment(self) -> None:
        if False:
            while True:
                i = 10
        node_with_multiple_objects_in_func = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u"\n                # Here we use object because stubs of protobuf are not\n                # available yet. So, instead of Any we used object here.\n                def foo(exp_id: object) -> object:\n                    return 'hi' #@\n                ")
        node_with_multiple_objects_in_func.file = filename
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_module(node_with_multiple_objects_in_func)
        temp_file.close()

    def test_raises_error_if_gap_between_type_and_comment_is_more_than_fifteen(self) -> None:
        if False:
            print('Hello World!')
        node_with_object_and_more_than_expected_gap = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u"\n                # Here we use object because stubs of protobuf are not\n                # available yet. So, instead of Any we used object here.\n\n                variable_one: str = '123'\n                variable_two: str = '1234'\n                # Some other content of module one.\n\n                # Line 1 content.\n                # Line 2 content.\n                # Line 3 content.\n                # Line 4 content.\n\n                # Some other content of module two.\n\n                def test_foo(arg: str) -> str:\n\n                def foo(exp_id: str) -> object:\n                    return 'hi' #@\n                ")
        node_with_object_and_more_than_expected_gap.file = filename
        message = testutils.MessageTest(msg_id='object-class-used', line=18, node=node_with_object_and_more_than_expected_gap)
        with self.checker_test_object.assertAddsMessages(message):
            self.checker_test_object.checker.visit_module(node_with_object_and_more_than_expected_gap)
        temp_file.close()
        node_with_object_and_less_than_fifteen_gap = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u"\n                # Here we use object because stubs of protobuf are not\n                # available yet. So, instead of Any we used object here.\n\n                variable_one: str = '123'\n                variable_two: str = '1234'\n                # Some other content of module one.\n\n                def test_foo(arg: str) -> str:\n\n                def foo(exp_id: str) -> object:\n                    return 'hi' #@\n                ")
        node_with_object_and_less_than_fifteen_gap.file = filename
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_module(node_with_object_and_less_than_fifteen_gap)
        temp_file.close()

    def test_no_error_raised_if_objects_are_present_with_todo_comment(self) -> None:
        if False:
            print('Hello World!')
        node_with_object_and_todo_comment = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u"\n                # TODO(#sll): Here we use object because stubs of protobuf\n                # are not available yet. So, instead of Any we used object\n                # here.\n                def foo(exp_id: object) -> object:\n                    return 'hi' #@\n                ")
        node_with_object_and_todo_comment.file = filename
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_module(node_with_object_and_todo_comment)
        temp_file.close()

class SingleLineCommentCheckerTests(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        super().setUp()
        self.checker_test_object = testutils.CheckerTestCase()
        self.checker_test_object.CHECKER_CLASS = pylint_extensions.SingleLineCommentChecker
        self.checker_test_object.setup_method()

    def test_invalid_punctuation(self) -> None:
        if False:
            i = 10
            return i + 15
        node_invalid_punctuation = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'# This is a multiline\n                # comment/\n\n                # Comment.\n                ')
        node_invalid_punctuation.file = filename
        node_invalid_punctuation.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_invalid_punctuation))
        message = testutils.MessageTest(msg_id='invalid-punctuation-used', line=2)
        with self.checker_test_object.assertAddsMessages(message):
            temp_file.close()

    def test_no_space_at_beginning(self) -> None:
        if False:
            return 10
        node_no_space_at_beginning = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'#Something.\n                ')
        node_no_space_at_beginning.file = filename
        node_no_space_at_beginning.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_no_space_at_beginning))
        message = testutils.MessageTest(msg_id='no-space-at-beginning', line=1)
        with self.checker_test_object.assertAddsMessages(message):
            temp_file.close()

    def test_no_capital_letter_at_beginning(self) -> None:
        if False:
            while True:
                i = 10
        node_no_capital_letter_at_beginning = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'# coding: utf-8\n\n                    # something.\n                ')
        node_no_capital_letter_at_beginning.file = filename
        node_no_capital_letter_at_beginning.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_no_capital_letter_at_beginning))
        message = testutils.MessageTest(msg_id='no-capital-letter-at-beginning', line=3)
        with self.checker_test_object.assertAddsMessages(message):
            temp_file.close()

    def test_comment_with_excluded_phrase(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        node_comment_with_excluded_phrase = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'# coding: utf-8\n                # pylint: disable\n                a = 1 + 2  # pylint: disable\n                ')
        node_comment_with_excluded_phrase.file = filename
        node_comment_with_excluded_phrase.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_comment_with_excluded_phrase))
        with self.checker_test_object.assertNoMessages():
            temp_file.close()

    def test_inline_comment_with_allowed_pragma_raises_no_error(self) -> None:
        if False:
            return 10
        node_inline_comment_with_allowed_pragma = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'a = 1 + 2  # type: ignore[some-rule]\n                ')
        node_inline_comment_with_allowed_pragma.file = filename
        node_inline_comment_with_allowed_pragma.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_inline_comment_with_allowed_pragma))
        with self.checker_test_object.assertNoMessages():
            temp_file.close()

    def test_inline_comment_with_multiple_allowed_pragmas_raises_no_error(self) -> None:
        if False:
            i = 10
            return i + 15
        node_inline_comment_with_allowed_pragma = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'a = 1 + 2  # isort:skip # pylint: ignore[some-rule]\n                ')
        node_inline_comment_with_allowed_pragma.file = filename
        node_inline_comment_with_allowed_pragma.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_inline_comment_with_allowed_pragma))
        with self.checker_test_object.assertNoMessages():
            temp_file.close()

    def test_inline_comment_with_invalid_pragma_raises_error(self) -> None:
        if False:
            print('Hello World!')
        node_inline_comment_with_invalid_pragma = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'a = 1 + 2  # not_a_valid_pragma\n                ')
        node_inline_comment_with_invalid_pragma.file = filename
        node_inline_comment_with_invalid_pragma.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_inline_comment_with_invalid_pragma))
        message = testutils.MessageTest(msg_id='no-allowed-inline-pragma', line=1)
        with self.checker_test_object.assertAddsMessages(message):
            temp_file.close()

    def test_variable_name_in_comment(self) -> None:
        if False:
            return 10
        node_variable_name_in_comment = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'# coding: utf-8\n\n                # variable_name is used.\n                ')
        node_variable_name_in_comment.file = filename
        node_variable_name_in_comment.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_variable_name_in_comment))
        with self.checker_test_object.assertNoMessages():
            temp_file.close()

    def test_comment_with_version_info(self) -> None:
        if False:
            i = 10
            return i + 15
        node_comment_with_version_info = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'# coding: utf-8\n\n                # v2 is used.\n                ')
        node_comment_with_version_info.file = filename
        node_comment_with_version_info.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_comment_with_version_info))
        with self.checker_test_object.assertNoMessages():
            temp_file.close()

    def test_data_type_in_comment(self) -> None:
        if False:
            return 10
        node_data_type_in_comment = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'# coding: utf-8\n\n                # str. variable is type of str.\n                ')
        node_data_type_in_comment.file = filename
        node_data_type_in_comment.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_data_type_in_comment))
        with self.checker_test_object.assertNoMessages():
            temp_file.close()

    def test_comment_inside_docstring(self) -> None:
        if False:
            while True:
                i = 10
        node_comment_inside_docstring = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'# coding: utf-8\n                    """# str. variable is type of str."""\n                    """# str. variable is type\n                    of str."""\n                ')
        node_comment_inside_docstring.file = filename
        node_comment_inside_docstring.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_comment_inside_docstring))
        with self.checker_test_object.assertNoMessages():
            temp_file.close()

    def test_well_formed_comment(self) -> None:
        if False:
            print('Hello World!')
        node_with_no_error_message = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'# coding: utf-8\n\n                # Multi\n                # line\n                # comment.\n                ')
        node_with_no_error_message.file = filename
        node_with_no_error_message.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_with_no_error_message))
        with self.checker_test_object.assertNoMessages():
            temp_file.close()

class BlankLineBelowFileOverviewCheckerTests(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            return 10
        super().setUp()
        self.checker_test_object = testutils.CheckerTestCase()
        self.checker_test_object.CHECKER_CLASS = pylint_extensions.BlankLineBelowFileOverviewChecker
        self.checker_test_object.setup_method()

    def test_no_empty_line_below_fileoverview(self) -> None:
        if False:
            return 10
        node_no_empty_line_below_fileoverview = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                    """ this file does something """\n                    import something\n                    import random\n                ')
        node_no_empty_line_below_fileoverview.file = filename
        node_no_empty_line_below_fileoverview.path = filename
        node_no_empty_line_below_fileoverview.fromlineno = 2
        self.checker_test_object.checker.visit_module(node_no_empty_line_below_fileoverview)
        message = testutils.MessageTest(msg_id='no-empty-line-provided-below-fileoverview', node=node_no_empty_line_below_fileoverview)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            temp_file.close()

    def test_extra_empty_lines_below_fileoverview(self) -> None:
        if False:
            while True:
                i = 10
        node_extra_empty_lines_below_fileoverview = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n\n                    """ this file does something """\n\n\n                    import something\n                    from something import random\n                ')
        node_extra_empty_lines_below_fileoverview.file = filename
        node_extra_empty_lines_below_fileoverview.path = filename
        node_extra_empty_lines_below_fileoverview.fromlineno = 2
        self.checker_test_object.checker.visit_module(node_extra_empty_lines_below_fileoverview)
        message = testutils.MessageTest(msg_id='only-a-single-empty-line-should-be-provided', node=node_extra_empty_lines_below_fileoverview)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            temp_file.close()

    def test_extra_empty_lines_below_fileoverview_with_unicode_characters(self) -> None:
        if False:
            return 10
        node_extra_empty_lines_below_fileoverview = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                    #this comment has a unicode character ✓\n                    """ this file does ✕ something """\n\n\n                    from something import random\n                ')
        node_extra_empty_lines_below_fileoverview.file = filename
        node_extra_empty_lines_below_fileoverview.path = filename
        node_extra_empty_lines_below_fileoverview.fromlineno = 3
        self.checker_test_object.checker.visit_module(node_extra_empty_lines_below_fileoverview)
        message = testutils.MessageTest(msg_id='only-a-single-empty-line-should-be-provided', node=node_extra_empty_lines_below_fileoverview)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            temp_file.close()

    def test_no_empty_line_below_fileoverview_with_unicode_characters(self) -> None:
        if False:
            print('Hello World!')
        node_no_empty_line_below_fileoverview = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                    #this comment has a unicode character ✓\n                    """ this file does ✕ something """\n                    import something\n                    import random\n                ')
        node_no_empty_line_below_fileoverview.file = filename
        node_no_empty_line_below_fileoverview.path = filename
        node_no_empty_line_below_fileoverview.fromlineno = 3
        self.checker_test_object.checker.visit_module(node_no_empty_line_below_fileoverview)
        message = testutils.MessageTest(msg_id='no-empty-line-provided-below-fileoverview', node=node_no_empty_line_below_fileoverview)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            temp_file.close()

    def test_single_new_line_below_file_overview(self) -> None:
        if False:
            print('Hello World!')
        node_with_no_error_message = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                    """ this file does something """\n\n                    import something\n                    import random\n                ')
        node_with_no_error_message.file = filename
        node_with_no_error_message.path = filename
        node_with_no_error_message.fromlineno = 2
        self.checker_test_object.checker.visit_module(node_with_no_error_message)
        with self.checker_test_object.assertNoMessages():
            temp_file.close()

    def test_file_with_no_file_overview(self) -> None:
        if False:
            return 10
        node_file_with_no_file_overview = astroid.scoped_nodes.Module(name='test', doc=None)
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                    import something\n                    import random\n                ')
        node_file_with_no_file_overview.file = filename
        node_file_with_no_file_overview.path = filename
        self.checker_test_object.checker.visit_module(node_file_with_no_file_overview)
        with self.checker_test_object.assertNoMessages():
            temp_file.close()

    def test_file_overview_at_end_of_file(self) -> None:
        if False:
            return 10
        node_file_overview_at_end_of_file = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                    """ this file does something """   ')
        node_file_overview_at_end_of_file.file = filename
        node_file_overview_at_end_of_file.path = filename
        node_file_overview_at_end_of_file.fromlineno = 2
        self.checker_test_object.checker.visit_module(node_file_overview_at_end_of_file)
        message = testutils.MessageTest(msg_id='only-a-single-empty-line-should-be-provided', node=node_file_overview_at_end_of_file)
        with self.checker_test_object.assertAddsMessages(message, ignore_position=True):
            temp_file.close()

class SingleLinePragmaCheckerTests(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        self.checker_test_object = testutils.CheckerTestCase()
        self.checker_test_object.CHECKER_CLASS = pylint_extensions.SingleLinePragmaChecker
        self.checker_test_object.setup_method()

    def test_pragma_for_multiline(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        node_pragma_for_multiline = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                    # pylint: disable=invalid-name\n                    def funcName():\n                        """ # pylint: disable=test-purpose"""\n                        pass\n                    # pylint: enable=invalid-name\n                ')
        node_pragma_for_multiline.file = filename
        node_pragma_for_multiline.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_pragma_for_multiline))
        message1 = testutils.MessageTest(msg_id='single-line-pragma', line=2)
        message2 = testutils.MessageTest(msg_id='single-line-pragma', line=6)
        with self.checker_test_object.assertAddsMessages(message1, message2):
            temp_file.close()

    def test_enable_single_line_pragma_for_multiline(self) -> None:
        if False:
            i = 10
            return i + 15
        node_enable_single_line_pragma_for_multiline = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                    # pylint: disable=single-line-pragma\n                    def func():\n                        """\n                        # pylint: disable=testing-purpose\n                        """\n                        pass\n                    # pylint: enable=single-line-pragma\n                ')
        node_enable_single_line_pragma_for_multiline.file = filename
        node_enable_single_line_pragma_for_multiline.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_enable_single_line_pragma_for_multiline))
        message = testutils.MessageTest(msg_id='single-line-pragma', line=2)
        with self.checker_test_object.assertAddsMessages(message):
            temp_file.close()

    def test_enable_single_line_pragma_with_invalid_name(self) -> None:
        if False:
            print('Hello World!')
        node_enable_single_line_pragma_with_invalid_name = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                    # pylint: disable=invalid-name, single-line-pragma\n                    def funcName():\n                        """\n                        # pylint: disable=testing-purpose\n                        """\n                        pass\n                    # pylint: enable=invalid_name, single-line-pragma\n                ')
        node_enable_single_line_pragma_with_invalid_name.file = filename
        node_enable_single_line_pragma_with_invalid_name.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_enable_single_line_pragma_with_invalid_name))
        message = testutils.MessageTest(msg_id='single-line-pragma', line=2)
        with self.checker_test_object.assertAddsMessages(message):
            temp_file.close()

    def test_single_line_pylint_pragma(self) -> None:
        if False:
            print('Hello World!')
        node_with_no_error_message = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                    def funcName():  # pylint: disable=single-line-pragma\n                        pass\n                ')
        node_with_no_error_message.file = filename
        node_with_no_error_message.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_with_no_error_message))
        with self.checker_test_object.assertNoMessages():
            temp_file.close()

    def test_no_and_extra_space_before_pylint(self) -> None:
        if False:
            print('Hello World!')
        node_no_and_extra_space_before_pylint = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                    # pylint:disable=single-line-pragma\n                    def func():\n                        """\n                        # pylint: disable=testing-purpose\n                        """\n                        pass\n                    # pylint:     enable=single-line-pragma\n                ')
        node_no_and_extra_space_before_pylint.file = filename
        node_no_and_extra_space_before_pylint.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_no_and_extra_space_before_pylint))
        message = testutils.MessageTest(msg_id='single-line-pragma', line=2)
        with self.checker_test_object.assertAddsMessages(message):
            temp_file.close()

class SingleSpaceAfterKeyWordCheckerTests(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.checker_test_object = testutils.CheckerTestCase()
        self.checker_test_object.CHECKER_CLASS = pylint_extensions.SingleSpaceAfterKeyWordChecker
        self.checker_test_object.setup_method()

    def test_no_space_after_keyword(self) -> None:
        if False:
            return 10
        node_no_space_after_keyword = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                if(False):\n                    pass\n                elif(True):\n                    pass\n                while(True):\n                    pass\n                yield(1)\n                return True if(True) else False\n                ')
        node_no_space_after_keyword.file = filename
        node_no_space_after_keyword.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_no_space_after_keyword))
        if_message = testutils.MessageTest(msg_id='single-space-after-keyword', args='if', line=2)
        elif_message = testutils.MessageTest(msg_id='single-space-after-keyword', args='elif', line=4)
        while_message = testutils.MessageTest(msg_id='single-space-after-keyword', args='while', line=6)
        yield_message = testutils.MessageTest(msg_id='single-space-after-keyword', args='yield', line=8)
        if_exp_message = testutils.MessageTest(msg_id='single-space-after-keyword', args='if', line=9)
        with self.checker_test_object.assertAddsMessages(if_message, elif_message, while_message, yield_message, if_exp_message):
            temp_file.close()

    def test_multiple_spaces_after_keyword(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        node_multiple_spaces_after_keyword = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                if  False:\n                    pass\n                elif  True:\n                    pass\n                while  True:\n                    pass\n                yield  1\n                return True if  True else False\n                ')
        node_multiple_spaces_after_keyword.file = filename
        node_multiple_spaces_after_keyword.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_multiple_spaces_after_keyword))
        if_message = testutils.MessageTest(msg_id='single-space-after-keyword', args='if', line=2)
        elif_message = testutils.MessageTest(msg_id='single-space-after-keyword', args='elif', line=4)
        while_message = testutils.MessageTest(msg_id='single-space-after-keyword', args='while', line=6)
        yield_message = testutils.MessageTest(msg_id='single-space-after-keyword', args='yield', line=8)
        if_exp_message = testutils.MessageTest(msg_id='single-space-after-keyword', args='if', line=9)
        with self.checker_test_object.assertAddsMessages(if_message, elif_message, while_message, yield_message, if_exp_message):
            temp_file.close()

    def test_single_space_after_keyword(self) -> None:
        if False:
            return 10
        node_single_space_after_keyword = astroid.scoped_nodes.Module(name='test', doc='Custom test')
        temp_file = tempfile.NamedTemporaryFile()
        filename = temp_file.name
        with utils.open_file(filename, 'w') as tmp:
            tmp.write(u'\n                if False:\n                    pass\n                elif True:\n                    pass\n                while True:\n                    pass\n                yield 1\n                return True if True else False\n                ')
        node_single_space_after_keyword.file = filename
        node_single_space_after_keyword.path = filename
        self.checker_test_object.checker.process_tokens(pylint_utils.tokenize_module(node_single_space_after_keyword))
        with self.checker_test_object.assertNoMessages():
            temp_file.close()

class InequalityWithNoneCheckerTests(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.checker_test_object = testutils.CheckerTestCase()
        self.checker_test_object.CHECKER_CLASS = pylint_extensions.InequalityWithNoneChecker
        self.checker_test_object.setup_method()

    def test_inequality_op_on_none_adds_message(self) -> None:
        if False:
            print('Hello World!')
        if_node = astroid.extract_node('\n            if x != None: #@\n                pass\n            ')
        compare_node = if_node.test
        not_equal_none_message = testutils.MessageTest(msg_id='inequality-with-none', node=compare_node)
        with self.checker_test_object.assertAddsMessages(not_equal_none_message, ignore_position=True):
            self.checker_test_object.checker.visit_compare(compare_node)

    def test_inequality_op_on_none_with_wrapped_none_adds_message(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if_node = astroid.extract_node('\n            if x != ( #@\n                None\n            ):\n                pass\n            ')
        compare_node = if_node.test
        not_equal_none_message = testutils.MessageTest(msg_id='inequality-with-none', node=compare_node)
        with self.checker_test_object.assertAddsMessages(not_equal_none_message, ignore_position=True):
            self.checker_test_object.checker.visit_compare(compare_node)

    def test_usage_of_is_not_on_none_does_not_add_message(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if_node = astroid.extract_node('\n            if x is not None: #@\n                pass\n            ')
        compare_node = if_node.test
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_compare(compare_node)

class DisallowedFunctionsCheckerTests(unittest.TestCase):
    """Unit tests for DisallowedFunctionsChecker"""

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        super().setUp()
        self.checker_test_object = testutils.CheckerTestCase()
        self.checker_test_object.CHECKER_CLASS = pylint_extensions.DisallowedFunctionsChecker
        self.checker_test_object.setup_method()

    def test_disallowed_removals_str(self) -> None:
        if False:
            print('Hello World!')
        self.checker_test_object.checker.config.disallowed_functions_and_replacements_str = ['example_func', 'a.example_attr']
        self.checker_test_object.checker.open()
        (call1, call2) = astroid.extract_node('\n        example_func() #@\n        a.example_attr() #@\n        ')
        message_remove_example_func = testutils.MessageTest(msg_id='remove-disallowed-function-calls', node=call1, args='example_func', confidence=interfaces.UNDEFINED)
        message_remove_example_attr = testutils.MessageTest(msg_id='remove-disallowed-function-calls', node=call2, args='a.example_attr', confidence=interfaces.UNDEFINED)
        with self.checker_test_object.assertAddsMessages(message_remove_example_func, message_remove_example_attr, ignore_position=True):
            self.checker_test_object.checker.visit_call(call1)
            self.checker_test_object.checker.visit_call(call2)

    def test_disallowed_replacements_str(self) -> None:
        if False:
            print('Hello World!')
        self.checker_test_object.checker.config.disallowed_functions_and_replacements_str = ['datetime.datetime.now=>datetime.datetime.utcnow', 'self.assertEquals=>self.assertEqual']
        self.checker_test_object.checker.open()
        (call1, call2, call3) = astroid.extract_node('\n            datetime.datetime.now() #@\n            self.assertEquals() #@\n            b.a.next() #@\n        ')
        message_replace_disallowed_datetime = testutils.MessageTest(msg_id='replace-disallowed-function-calls', node=call1, args=('datetime.datetime.now', 'datetime.datetime.utcnow'), confidence=interfaces.UNDEFINED)
        message_replace_disallowed_assert_equals = testutils.MessageTest(msg_id='replace-disallowed-function-calls', node=call2, args=('self.assertEquals', 'self.assertEqual'), confidence=interfaces.UNDEFINED)
        with self.checker_test_object.assertAddsMessages(message_replace_disallowed_datetime, message_replace_disallowed_assert_equals, ignore_position=True):
            self.checker_test_object.checker.visit_call(call1)
            self.checker_test_object.checker.visit_call(call2)
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_call(call3)

    def test_disallowed_removals_regex(self) -> None:
        if False:
            i = 10
            return i + 15
        self.checker_test_object.checker.config.disallowed_functions_and_replacements_regex = ['.*example_func', '.*\\..*example_attr']
        self.checker_test_object.checker.open()
        (call1, call2) = astroid.extract_node('\n        somethingexample_func() #@\n        c.someexample_attr() #@\n        ')
        message_remove_example_func = testutils.MessageTest(msg_id='remove-disallowed-function-calls', node=call1, args='somethingexample_func', confidence=interfaces.UNDEFINED)
        message_remove_example_attr = testutils.MessageTest(msg_id='remove-disallowed-function-calls', node=call2, args='c.someexample_attr', confidence=interfaces.UNDEFINED)
        with self.checker_test_object.assertAddsMessages(message_remove_example_func, message_remove_example_attr, ignore_position=True):
            self.checker_test_object.checker.visit_call(call1)
            self.checker_test_object.checker.visit_call(call2)

    def test_disallowed_replacements_regex(self) -> None:
        if False:
            return 10
        self.checker_test_object.checker.config.disallowed_functions_and_replacements_regex = ['.*example_func=>other_func', '.*\\.example_attr=>other_attr']
        self.checker_test_object.checker.open()
        (call1, call2, call3, call4) = astroid.extract_node('\n        somethingexample_func() #@\n        d.example_attr() #@\n        d.example_attr() #@\n        d.b.example_attr() #@\n        ')
        message_replace_example_func = testutils.MessageTest(msg_id='replace-disallowed-function-calls', node=call1, args=('somethingexample_func', 'other_func'), confidence=interfaces.UNDEFINED)
        message_replace_example_attr1 = testutils.MessageTest(msg_id='replace-disallowed-function-calls', node=call2, args=('d.example_attr', 'other_attr'), confidence=interfaces.UNDEFINED)
        message_replace_example_attr2 = testutils.MessageTest(msg_id='replace-disallowed-function-calls', node=call3, args=('d.example_attr', 'other_attr'), confidence=interfaces.UNDEFINED)
        message_replace_example_attr3 = testutils.MessageTest(msg_id='replace-disallowed-function-calls', node=call4, args=('d.b.example_attr', 'other_attr'), confidence=interfaces.UNDEFINED)
        with self.checker_test_object.assertAddsMessages(message_replace_example_func, message_replace_example_attr1, message_replace_example_attr2, message_replace_example_attr3, ignore_position=True):
            self.checker_test_object.checker.visit_call(call1)
            self.checker_test_object.checker.visit_call(call2)
            self.checker_test_object.checker.visit_call(call3)
            self.checker_test_object.checker.visit_call(call4)

class NonTestFilesFunctionNameCheckerTests(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            return 10
        super().setUp()
        self.checker_test_object = testutils.CheckerTestCase()
        self.checker_test_object.CHECKER_CLASS = pylint_extensions.NonTestFilesFunctionNameChecker
        self.checker_test_object.setup_method()

    def test_function_def_for_test_file_with_test_only_adds_no_msg(self) -> None:
        if False:
            print('Hello World!')
        def_node = astroid.extract_node('\n            def test_only_some_random_function(param1, param2):\n                pass\n            ')
        def_node.root().name = 'random_module_test'
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_functiondef(def_node)

    def test_function_def_for_test_file_without_test_only_adds_no_msg(self) -> None:
        if False:
            while True:
                i = 10
        def_node = astroid.extract_node('\n            def some_random_function(param1, param2):\n                pass\n            ')
        def_node.root().name = 'random_module_test'
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_functiondef(def_node)

    def test_function_def_for_non_test_file_with_test_only_adds_msg(self) -> None:
        if False:
            return 10
        def_node = astroid.extract_node('\n            def test_only_some_random_function(param1, param2):\n                pass\n            ')
        def_node.root().name = 'random_module_nontest'
        non_test_function_name_message = testutils.MessageTest(msg_id='non-test-files-function-name-checker', node=def_node)
        with self.checker_test_object.assertAddsMessages(non_test_function_name_message, ignore_position=True):
            self.checker_test_object.checker.visit_functiondef(def_node)

    def test_function_def_for_non_test_file_without_test_only_adds_no_msg(self) -> None:
        if False:
            i = 10
            return i + 15
        def_node = astroid.extract_node('\n            def some_random_function(param1, param2):\n                pass\n            ')
        def_node.root().name = 'random_module_nontest'
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_functiondef(def_node)

class DisallowHandlerWithoutSchemaTests(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        self.checker_test_object = testutils.CheckerTestCase()
        self.checker_test_object.CHECKER_CLASS = pylint_extensions.DisallowHandlerWithoutSchema
        self.checker_test_object.setup_method()

    def test_schema_handlers_without_request_args_raise_error(self) -> None:
        if False:
            return 10
        schemaless_class_node = astroid.extract_node('\n            class BaseHandler():\n                HANDLER_ARGS_SCHEMAS = None\n                URL_PATH_ARGS_SCHEMAS = None\n\n            class FakeClass(BaseHandler):\n                URL_PATH_ARGS_SCHEMAS = {}\n            ')
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='no-schema-for-handler-args', node=schemaless_class_node, args=schemaless_class_node.name), ignore_position=True):
            self.checker_test_object.checker.visit_classdef(schemaless_class_node)

    def test_schema_handlers_without_url_path_args_raise_error(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        schemaless_class_node = astroid.extract_node('\n            class BaseHandler():\n                HANDLER_ARGS_SCHEMAS = None\n                URL_PATH_ARGS_SCHEMAS = None\n\n            class FakeClass(BaseHandler):\n                HANDLER_ARGS_SCHEMAS = {}\n            ')
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='no-schema-for-url-path-elements', node=schemaless_class_node, args=schemaless_class_node.name), ignore_position=True):
            self.checker_test_object.checker.visit_classdef(schemaless_class_node)

    def test_handlers_with_valid_schema_do_not_raise_error(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        schemaless_class_node = astroid.extract_node("\n            class BaseHandler():\n                HANDLER_ARGS_SCHEMAS = None\n                URL_PATH_ARGS_SCHEMAS = None\n\n            class FakeClass(BaseHandler):\n                URL_PATH_ARGS_SCHEMAS = {}\n                HANDLER_ARGS_SCHEMAS = {'GET': {}}\n            ")
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_classdef(schemaless_class_node)

    def test_list_of_non_schema_handlers_do_not_raise_errors(self) -> None:
        if False:
            while True:
                i = 10
        'Handler class name in list of\n        HANDLER_CLASS_NAMES_WHICH_STILL_NEED_SCHEMAS do not raise error\n        for missing schemas.\n        '
        schemaless_class_node = astroid.extract_node('\n            class BaseHandler():\n                HANDLER_ARGS_SCHEMAS = None\n                URL_PATH_ARGS_SCHEMAS = None\n\n            class SessionBeginHandler(BaseHandler):\n                def get(self):\n                    return\n            ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_classdef(schemaless_class_node)

    def test_schema_handler_with_basehandler_as_an_ancestor_raise_error(self) -> None:
        if False:
            return 10
        'Handlers which are child classes of BaseHandler must have schema\n        defined locally in the class.\n        '
        schemaless_class_node = astroid.extract_node('\n            class BaseHandler():\n                HANDLER_ARGS_SCHEMAS = None\n                URL_PATH_ARGS_SCHEMAS = None\n\n            class BaseClass(BaseHandler):\n                HANDLER_ARGS_SCHEMAS = {}\n                URL_PATH_ARGS_SCHEMAS = {}\n\n            class FakeClass(BaseClass):\n                HANDLER_ARGS_SCHEMAS = {}\n            ')
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='no-schema-for-url-path-elements', node=schemaless_class_node, args=schemaless_class_node.name), ignore_position=True):
            self.checker_test_object.checker.visit_classdef(schemaless_class_node)

    def test_wrong_data_type_in_url_path_args_schema_raise_error(self) -> None:
        if False:
            while True:
                i = 10
        'Checks whether the schemas in URL_PATH_ARGS_SCHEMAS must be of\n        dict type.\n        '
        schemaless_class_node = astroid.extract_node('\n            class BaseHandler():\n                HANDLER_ARGS_SCHEMAS = None\n                URL_PATH_ARGS_SCHEMA = None\n\n            class BaseClass(BaseHandler):\n                HANDLER_ARGS_SCHEMAS = {}\n                URL_PATH_ARGS_SCHEMAS = {}\n\n            class FakeClass(BaseClass):\n                URL_PATH_ARGS_SCHEMAS = 5\n                HANDLER_ARGS_SCHEMAS = {}\n            ')
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='url-path-args-schemas-must-be-dict', node=schemaless_class_node, args=schemaless_class_node.name), ignore_position=True):
            self.checker_test_object.checker.visit_classdef(schemaless_class_node)

    def test_wrong_data_type_in_handler_args_schema_raise_error(self) -> None:
        if False:
            return 10
        'Checks whether the schemas in URL_PATH_ARGS_SCHEMAS must be of\n        dict type.\n        '
        schemaless_class_node = astroid.extract_node('\n            class BaseHandler():\n                HANDLER_ARGS_SCHEMAS = None\n                URL_PATH_ARGS_SCHEMAS = None\n\n            class BaseClass(BaseHandler):\n                HANDLER_ARGS_SCHEMAS = {}\n                URL_PATH_ARGS_SCHEMAS = {}\n\n            class FakeClass(BaseClass):\n                URL_PATH_ARGS_SCHEMAS = {}\n                HANDLER_ARGS_SCHEMAS = 10\n            ')
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='handler-args-schemas-must-be-dict', node=schemaless_class_node, args=schemaless_class_node.name), ignore_position=True):
            self.checker_test_object.checker.visit_classdef(schemaless_class_node)

class DisallowedImportsCheckerTests(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        self.checker_test_object = testutils.CheckerTestCase()
        self.checker_test_object.CHECKER_CLASS = pylint_extensions.DisallowedImportsChecker
        self.checker_test_object.setup_method()

    def test_importing_text_from_typing_in_single_line_raises_error(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        node = astroid.extract_node('from typing import Any, cast, Text')
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='disallowed-text-import', node=node), ignore_position=True):
            self.checker_test_object.checker.visit_importfrom(node)

    def test_importing_text_from_typing_in_multi_line_raises_error(self) -> None:
        if False:
            while True:
                i = 10
        node = astroid.extract_node('\n            from typing import (\n                Any, Dict, List, Optional, Sequence, Text, TypeVar)\n            ')
        with self.checker_test_object.assertAddsMessages(testutils.MessageTest(msg_id='disallowed-text-import', node=node), ignore_position=True):
            self.checker_test_object.checker.visit_importfrom(node)

    def test_non_import_of_text_from_typing_does_not_raise_error(self) -> None:
        if False:
            i = 10
            return i + 15
        node = astroid.extract_node('\n            from typing import Any, Dict, List, Optional\n            ')
        with self.checker_test_object.assertNoMessages():
            self.checker_test_object.checker.visit_importfrom(node)