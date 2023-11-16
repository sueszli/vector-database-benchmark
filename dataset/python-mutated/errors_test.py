import json
import textwrap
import unittest
from typing import Dict, List, Optional, Tuple
from unittest.mock import call, patch
from .. import errors, UserError
from ..ast import UnstableAST
from ..errors import _get_unused_ignore_codes, _line_ranges_spanned_by_format_strings, _map_line_to_start_of_range, _relocate_errors, _remove_unused_ignores, _suppress_errors, Errors, LineBreakParsingException, PartialErrorSuppression, SkippingGeneratedFileException
unittest.util._MAX_LENGTH = 200

def _normalize(input: str) -> str:
    if False:
        print('Hello World!')
    return textwrap.dedent(input).strip().replace('FIXME', 'pyre-fixme')

class ErrorsTest(unittest.TestCase):

    def test_from_json(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(Errors.from_json('[{ "path": "test.py", "key": "value" }]'), Errors([{'path': 'test.py', 'key': 'value'}]))
        with patch('sys.stdin.read', return_value='[{ "path": "test.py", "key": "value" }]'):
            self.assertEqual(Errors.from_stdin(), Errors([{'path': 'test.py', 'key': 'value'}]))
        self.assertEqual(Errors.from_json(json.dumps([{'path': 'test.py', 'key': 'value', 'code': 1}, {'path': 'test.py', 'key': 'value', 'code': 2}]), only_fix_error_code=1), Errors([{'path': 'test.py', 'key': 'value', 'code': 1}]))
        with patch('sys.stdin.read', return_value=json.dumps([{'path': 'test.py', 'key': 'value', 'code': 1}, {'path': 'test.py', 'key': 'value', 'code': 2}])):
            self.assertEqual(Errors.from_stdin(only_fix_error_code=1), Errors([{'path': 'test.py', 'key': 'value', 'code': 1}]))
        with self.assertRaises(UserError):
            Errors.from_json('[{ "path": "test.py", "key": "value" }')

    def test_paths_to_errors(self) -> None:
        if False:
            print('Hello World!')
        errors = Errors([{'path': 'test1.py', 'key': 'value', 'code': 1}, {'path': 'test2.py', 'key': 'value', 'code': 2}, {'path': 'test1.py', 'key': 'value', 'code': 3}])
        self.assertEqual(errors.paths_to_errors, {'test1.py': [{'code': 1, 'key': 'value', 'path': 'test1.py'}, {'code': 3, 'key': 'value', 'path': 'test1.py'}], 'test2.py': [{'code': 2, 'key': 'value', 'path': 'test2.py'}]})

    @patch.object(errors.Path, 'read_text', return_value='')
    @patch.object(errors.Path, 'write_text')
    def test_suppress(self, path_write_text, path_read_text) -> None:
        if False:
            i = 10
            return i + 15
        with patch(f'{errors.__name__}._suppress_errors', return_value='<transformed>'):
            Errors([{'path': 'path.py', 'line': 1, 'concise_description': 'Error [1]: description'}, {'path': 'other.py', 'line': 2, 'concise_description': 'Error [2]: description'}]).suppress()
            path_read_text.assert_has_calls([call(), call()])
            path_write_text.assert_has_calls([call('<transformed>'), call('<transformed>')])
        with patch(f'{errors.__name__}._suppress_errors', side_effect=UnstableAST()):
            with self.assertRaises(PartialErrorSuppression) as context:
                Errors([{'path': 'path.py', 'line': 1, 'concise_description': 'Error [1]: description'}, {'path': 'other.py', 'line': 2, 'concise_description': 'Error [2]: description'}]).suppress()
            self.assertEqual(set(context.exception.unsuppressed_paths), {'path.py', 'other.py'})

    def test_get_unused_ignore_codes(self) -> None:
        if False:
            return 10
        self.assertEqual(_get_unused_ignore_codes([{'code': '0', 'description': 'The `pyre-ignore[1, 9]` or `pyre-fixme[1, 9]` ' + 'comment is not suppressing type errors, please remove it.'}]), [1, 9])
        self.assertEqual(_get_unused_ignore_codes([{'code': '0', 'description': 'The `pyre-ignore[1, 9]` or `pyre-fixme[1, 9]` ' + 'comment is not suppressing type errors, please remove it.'}, {'code': '0', 'description': 'The `pyre-ignore[2]` or `pyre-fixme[2]` ' + 'comment is not suppressing type errors, please remove it.'}]), [1, 2, 9])
        self.assertEqual(_get_unused_ignore_codes([{'code': '1', 'description': 'The `pyre-ignore[1, 9]` or `pyre-fixme[1, 9]` ' + 'comment is not suppressing type errors, please remove it.'}]), [])
        self.assertEqual(_get_unused_ignore_codes([{'code': '1', 'description': 'The `pyre-ignore[]` or `pyre-fixme[]` ' + 'comment is not suppressing type errors, please remove it.'}]), [])

    @patch.object(errors, '_get_unused_ignore_codes')
    def test_remove_unused_ignores(self, get_unused_ignore_codes) -> None:
        if False:
            print('Hello World!')
        get_unused_ignore_codes.return_value = [1, 3, 4]
        self.assertEqual(_remove_unused_ignores('# pyre-fixme[1, 2, 3, 4]: Comment', []), '# pyre-fixme[2]: Comment')
        get_unused_ignore_codes.return_value = [1, 2, 3, 4]
        self.assertEqual(_remove_unused_ignores('# pyre-fixme[1, 2, 3, 4]: Comment', []), '')
        get_unused_ignore_codes.return_value = [1]
        self.assertEqual(_remove_unused_ignores('#  pyre-fixme[1]: Comment', []), '')
        get_unused_ignore_codes.return_value = [1]
        self.assertEqual(_remove_unused_ignores('# pyre-fixme[2, 3, 4]: Comment', []), '# pyre-fixme[2, 3, 4]: Comment')
        get_unused_ignore_codes.return_value = [1, 2]
        self.assertEqual(_remove_unused_ignores('# pyre-fixme: Comment', []), '')
        get_unused_ignore_codes.return_value = [1, 2]
        self.assertEqual(_remove_unused_ignores('# Unrelated comment. # pyre-fixme[1, 2]: Comment', []), '# Unrelated comment.')
        get_unused_ignore_codes.return_value = [1, 3, 4]
        self.assertEqual(_remove_unused_ignores('# pyre-fixme    [1, 2, 3, 4]: Comment', []), '# pyre-fixme    [2]: Comment')

    def assertSuppressErrors(self, errors: Dict[int, List[Dict[str, str]]], input: str, expected_output: str, *, custom_comment: Optional[str]=None, max_line_length: Optional[int]=None, truncate: bool=False, unsafe: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(_suppress_errors(_normalize(input), errors, custom_comment, max_line_length, truncate, unsafe), _normalize(expected_output))

    def test_suppress_errors(self) -> None:
        if False:
            return 10
        self.assertSuppressErrors({}, '\n            def foo() -> None: pass\n            ', '\n            def foo() -> None: pass\n            ')
        self.assertSuppressErrors({1: [{'code': '1', 'description': 'description'}]}, '\n            def foo() -> None: pass\n            ', '\n            # FIXME[1]: description\n            def foo() -> None: pass\n            ')
        self.assertSuppressErrors({1: [{'code': '1', 'description': 'description 1'}, {'code': '2', 'description': 'description duplicate'}, {'code': '2', 'description': 'description duplicate'}, {'code': '1', 'description': 'description 2'}]}, '\n            def foo() -> None: pass\n            ', '\n            # FIXME[1]: description 1\n            # FIXME[2]: description duplicate\n            # FIXME[1]: description 2\n            def foo() -> None: pass\n            ')
        self.assertSuppressErrors({2: [{'code': '1', 'description': 'description'}]}, '\n            def foo() -> None:\n                pass\n            ', '\n            def foo() -> None:\n                # FIXME[1]: description\n                pass\n            ')
        self.assertSuppressErrors({1: [{'code': '404', 'description': 'description'}]}, '\n            # this is an unparseable file\n\n            def foo()\n                pass\n            ', '\n            # this is an unparseable file\n            # pyre-ignore-all-errors[404]\n\n            def foo()\n                pass\n            ')
        with self.assertRaises(SkippingGeneratedFileException):
            _suppress_errors('# @generated', {})
        with self.assertRaises(LineBreakParsingException):
            _suppress_errors(_normalize('\n                    def foo() -> None:\n                        line_break = \\\n                            [\n                                param\n                            ]\n                        unrelated_line = 0\n                    '), {3: [{'code': '1', 'description': 'description'}]})
        try:
            _suppress_errors('# @generated', {}, custom_comment=None, max_line_length=None, truncate=False, unsafe=True)
        except SkippingGeneratedFileException:
            self.fail('Unexpected `SkippingGeneratedFileException` exception.')
        self.assertSuppressErrors({1: [{'code': '1', 'description': 'description'}]}, '\n            def foo() -> None: pass\n            ', '\n            # FIXME[1]: T1234\n            def foo() -> None: pass\n            ', custom_comment='T1234')
        self.assertSuppressErrors({2: [{'code': '1', 'description': 'description'}]}, '\n            # comment\n            def foo() -> None: pass\n            ', '\n            # comment\n            # FIXME[1]: description\n            def foo() -> None: pass\n            ')
        self.assertSuppressErrors({1: [{'code': '0', 'description': 'description'}]}, '\n            def foo() -> None: # FIXME[1]\n                # comment\n                pass\n            ', '\n            def foo() -> None:\n                # comment\n                pass\n            ')
        self.assertSuppressErrors({1: [{'code': '1', 'description': 'description'}], 2: [{'code': '2', 'description': 'description'}]}, '\n            def foo() -> None:\n                pass\n            ', '\n            # FIXME[1]: description\n            def foo() -> None:\n                # FIXME[2]: description\n                pass\n            ')
        self.assertSuppressErrors({1: [{'code': '1', 'description': 'description'}, {'code': '2', 'description': 'description'}]}, '\n            def foo() -> None: pass\n            ', '\n            # FIXME[1]: description\n            # FIXME[2]: description\n            def foo() -> None: pass\n            ')
        self.assertSuppressErrors({1: [{'code': '1', 'description': 'description'}]}, '\n            def foo() -> None: pass\n            ', '\n            # FIXME[1]:\n            #  description\n            def foo() -> None: pass\n            ', max_line_length=20)
        self.assertSuppressErrors({1: [{'code': '1', 'description': 'description'}]}, '\n            def foo() -> None: pass\n            ', '\n            # FIXME[1]: descr...\n            def foo() -> None: pass\n            ', max_line_length=25, truncate=True)
        self.assertSuppressErrors({1: [{'code': '1', 'description': 'this description takes up over four lines                         of content when it is split, given the max line length'}]}, '\n            def foo() -> None: pass\n            ', '\n            # FIXME[1]: this ...\n            def foo() -> None: pass\n            ', max_line_length=25)

    def test_suppress_errors__remove_unused(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertSuppressErrors({1: [{'code': '0', 'description': 'description'}]}, '\n            # FIXME[0]: ignore\n            def foo() -> None: pass\n            ', '\n            def foo() -> None: pass\n            ')
        self.assertSuppressErrors({1: [{'code': '0', 'description': 'description'}]}, '\n            # FIXME[0]: ignore\n            #  over multple lines\n            def foo() -> None: pass\n            ', '\n            def foo() -> None: pass\n            ')
        self.assertSuppressErrors({1: [{'code': '0', 'description': 'description'}]}, '\n            # FIXME[0]: ignore\n            #  over multple lines\n            # FIXME[1]: description\n            def foo() -> None: pass\n            ', '\n            # FIXME[1]: description\n            def foo() -> None: pass\n            ')
        self.assertSuppressErrors({1: [{'code': '0', 'description': 'description'}]}, '\n            def foo() -> None: pass  # FIXME[0]: ignore\n            ', '\n            def foo() -> None: pass\n            ')
        self.assertSuppressErrors({1: [{'code': '0', 'description': 'description'}], 2: [{'code': '0', 'description': 'description'}]}, '\n            # FIXME[1]: ignore\n            # FIXME[2]: ignore\n            def foo() -> None: pass\n            ', '\n            def foo() -> None: pass\n            ')
        self.assertSuppressErrors({1: [{'code': '0', 'description': 'description'}, {'code': '2', 'description': 'new error'}]}, '\n            def foo() -> None: pass  # FIXME[1]\n            ', '\n            # FIXME[2]: new error\n            def foo() -> None: pass\n            ')
        self.assertSuppressErrors({1: [{'code': '2', 'description': 'new error'}, {'code': '0', 'description': 'description'}]}, '\n            def foo() -> None: pass  # FIXME[1]\n            ', '\n            # FIXME[2]: new error\n            def foo() -> None: pass\n            ')
        self.assertSuppressErrors({1: [{'code': '0', 'description': 'The `pyre-ignore[1]` or `pyre-fixme[1]` ' + 'comment is not suppressing type errors, please remove it.'}]}, '\n            def foo() -> None: pass  # FIXME[1, 2]\n            ', '\n            def foo() -> None: pass  # FIXME[2]\n            ')
        self.assertSuppressErrors({1: [{'code': '0', 'description': 'The `pyre-ignore[1, 3]` or `pyre-fixme[1, 3]` ' + 'comment is not suppressing type errors, please remove it.'}]}, '\n            # FIXME[1, 2, 3]\n            # Continuation comment.\n            def foo() -> None: pass\n            ', '\n            # FIXME[2]\n            # Continuation comment.\n            def foo() -> None: pass\n            ')
        self.assertSuppressErrors({1: [{'code': '0', 'description': 'The `pyre-ignore[1, 3]` or `pyre-fixme[1, 3]` ' + 'comment is not suppressing type errors, please remove it.'}]}, '\n            # FIXME[1, 2, 3]: Comment[Comment]\n            def foo() -> None: pass\n            ', '\n            # FIXME[2]: Comment[Comment]\n            def foo() -> None: pass\n            ')
        self.assertSuppressErrors({1: [{'code': '0', 'description': 'The `pyre-ignore[1, 3]` or `pyre-fixme[1, 3]` ' + 'comment is not suppressing type errors, please remove it.'}]}, '\n            # FIXME[1, 3]\n            # Continuation comment.\n            def foo() -> None: pass\n            ', '\n            def foo() -> None: pass\n            ')
        self.assertSuppressErrors({1: [{'code': '0', 'description': 'The `pyre-ignore[1, 3]` or `pyre-fixme[1, 3]` ' + 'comment is not suppressing type errors, please remove it.'}], 2: [{'code': '4', 'description': 'Description.'}]}, '\n            # FIXME[1, 2, 3]\n            def foo() -> None: pass\n            ', '\n            # FIXME[2]\n            # FIXME[4]: Description.\n            def foo() -> None: pass\n            ')

    def test_suppress_errors__line_breaks(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertSuppressErrors({}, '\n            def foo() -> None:\n                """\n                Random line break that won\'t parse.\n                /!\\\n                Text.\n                """\n                pass\n            ', '\n            def foo() -> None:\n                """\n                Random line break that won\'t parse.\n                /!\\\n                Text.\n                """\n                pass\n            ')
        self.assertSuppressErrors({3: [{'code': '1', 'description': 'description'}], 4: [{'code': '2', 'description': 'description'}]}, '\n            def foo() -> None:\n                x = something + \\\n                        error() + \\\n                        error()  # unrelated comment\n            ', '\n            def foo() -> None:\n                x = (something +\n                        # FIXME[1]: description\n                        error() +\n                        # FIXME[2]: description\n                        error())  # unrelated comment\n            ')
        self.assertSuppressErrors({3: [{'code': '1', 'description': 'description'}]}, '\n            def foo() -> None:\n                x, y, z = \\\n                    error()\n            ', '\n            def foo() -> None:\n                x, y, z = (\n                    # FIXME[1]: description\n                    error())\n            ')
        self.assertSuppressErrors({3: [{'code': '1', 'description': 'description'}]}, '\n            def foo() -> None:\n                x: int = \\\n                    1\n            ', '\n            def foo() -> None:\n                x: int = (\n                    # FIXME[1]: description\n                    1)\n            ')
        self.assertSuppressErrors({3: [{'code': '1', 'description': 'description'}]}, '\n            def foo() -> None:\n                del \\\n                    error\n            ', '\n            def foo() -> None:\n                del (\n                    # FIXME[1]: description\n                    error)\n            ')
        self.assertSuppressErrors({3: [{'code': '1', 'description': 'description'}]}, '\n            def foo() -> None:\n                assert \\\n                    test\n            ', '\n            def foo() -> None:\n                assert (\n                    # FIXME[1]: description\n                    test)\n            ')
        self.assertSuppressErrors({3: [{'code': '1', 'description': 'description'}]}, '\n            def foo() -> None:\n                assert test + \\\n                    test2\n            ', '\n            def foo() -> None:\n                assert (test +\n                    # FIXME[1]: description\n                    test2)\n            ')
        self.assertSuppressErrors({3: [{'code': '1', 'description': 'description'}]}, '\n            def foo() -> None:\n                assert test, \\\n                    "message"\n            ', '\n            def foo() -> None:\n                assert (test), (\n                    # FIXME[1]: description\n                    "message")\n            ')
        self.assertSuppressErrors({3: [{'code': '1', 'description': 'description'}]}, '\n            def foo() -> None:\n                raise \\\n                    Exception()\n            ', '\n            def foo() -> None:\n                raise (\n                    # FIXME[1]: description\n                    Exception())\n            ')
        self.assertSuppressErrors({3: [{'code': '1', 'description': 'description'}]}, '\n            def foo() -> None:\n                return a + \\\n                    error\n            ', '\n            def foo() -> None:\n                return (a +\n                    # FIXME[1]: description\n                    error)\n            ')
        self.assertSuppressErrors({3: [{'code': '1', 'description': 'description'}]}, '\n            def foo() -> None:\n                return \\\n                    error\n            ', '\n            def foo() -> None:\n                return (\n                    # FIXME[1]: description\n                    error)\n            ')
        self.assertSuppressErrors({3: [{'code': '1', 'description': 'description'}]}, '\n            def foo() -> None:\n                line_break = \\\n                    trailing_open(\n                        param)\n                unrelated_line = 1\n            ', '\n            def foo() -> None:\n                line_break = (\n                    # FIXME[1]: description\n                    trailing_open(\n                        param))\n                unrelated_line = 1\n            ')
        self.assertSuppressErrors({3: [{'code': '1', 'description': 'description'}]}, '\n            def foo() -> None:\n                line_break = \\\n                    trailing_open(\n                        param1,\n                        param2,\n                    )\n                unrelated_line = 1\n            ', '\n            def foo() -> None:\n                line_break = (\n                    # FIXME[1]: description\n                    trailing_open(\n                        param1,\n                        param2,\n                    ))\n                unrelated_line = 1\n            ')

    def test_suppress_errors__long_class_name(self) -> None:
        if False:
            return 10
        self.assertSuppressErrors({1: [{'code': '1', 'description': 'This is a                         really.long.class.name.exceeding.twenty.five.Characters'}]}, '\n            def foo() -> None: pass\n            ', '\n            # FIXME[1]: This ...\n            def foo() -> None: pass\n            ', max_line_length=25)

    def test_suppress_errors__manual_import(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertSuppressErrors({3: [{'code': '21', 'description': 'description'}], 4: [{'code': '21', 'description': 'description'}]}, '\n            from a import b\n            # @manual=//special:case\n            from a import c\n            from a import d\n            ', '\n            from a import b\n            # FIXME[21]: description\n            # @manual=//special:case\n            from a import c\n            # FIXME[21]: description\n            from a import d\n            ')

    def test_suppress_errors__multi_line_string(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertSuppressErrors({5: [{'code': '1', 'description': 'description'}]}, '\n            def foo() -> None:\n                call("""\\\n                    some text\n                    more text\n                    """, problem_arg)\n            ', '\n            def foo() -> None:\n                call("""\\\n                    some text\n                    more text\n                    """, problem_arg)  # FIXME[1]\n            ')
        self.assertSuppressErrors({5: [{'code': '1', 'description': 'description'}, {'code': '2', 'description': 'description'}]}, '\n            def foo() -> None:\n                x = ("""\\\n                some text\n                more text\n                """, problem)\n            ', '\n            def foo() -> None:\n                x = ("""\\\n                some text\n                more text\n                """, problem)  # FIXME[1, 2]\n            ')
        self.assertSuppressErrors({5: [{'code': '0', 'description': 'unused ignore'}]}, '\n            def foo() -> None:\n                call("""\\\n                    some text\n                    more text\n                    """, problem_arg)  # FIXME[1]\n            ', '\n            def foo() -> None:\n                call("""\\\n                    some text\n                    more text\n                    """, problem_arg)\n            ')

    def test_suppress_errors__format_string(self) -> None:
        if False:
            return 10
        self.assertSuppressErrors({4: [{'code': '42', 'description': 'Some error'}], 5: [{'code': '42', 'description': 'Some error'}, {'code': '43', 'description': 'Some error'}]}, '\n            def foo() -> None:\n                f"""\n                foo\n                {1 + "hello"}\n                {"world" + int("a")}\n                bar\n                """\n            ', '\n            def foo() -> None:\n                # FIXME[42]: Some error\n                # FIXME[43]: Some error\n                f"""\n                foo\n                {1 + "hello"}\n                {"world" + int("a")}\n                bar\n                """\n            ')
        self.assertSuppressErrors({4: [{'code': '42', 'description': 'Some error 1'}, {'code': '42', 'description': 'Some error 2'}]}, '\n            def foo() -> None:\n                f"""\n                foo\n                {1 + "hello"}\n                {"world" + int("a")}\n                bar\n                """\n            ', '\n            def foo() -> None:\n                # FIXME[42]: Some error 1\n                # FIXME[42]: Some error 2\n                f"""\n                foo\n                {1 + "hello"}\n                {"world" + int("a")}\n                bar\n                """\n            ')

    def test_suppress_errors__empty_fixme_code(self) -> None:
        if False:
            print('Hello World!')
        self.assertSuppressErrors({2: [{'code': '0', 'description': 'Some error'}]}, '\n            def foo() -> None:\n                # FIXME[]\n                unused_ignore: str = "hello"\n            ', '\n            def foo() -> None:\n                unused_ignore: str = "hello"\n            ')
        self.assertSuppressErrors({2: [{'code': '0', 'description': 'Some error'}], 3: [{'code': '42', 'description': 'Some error'}]}, '\n            def foo() -> None:\n                # FIXME[]\n                x: str = 1\n            ', '\n            def foo() -> None:\n                # FIXME[42]: Some error\n                x: str = 1\n            ')
        self.assertSuppressErrors({2: [{'code': '0', 'description': 'Some error'}]}, '\n            def foo() -> None:\n                # FIXME[,]\n                unused_ignore: str = "hello"\n            ', '\n            def foo() -> None:\n                unused_ignore: str = "hello"\n            ')

    def assertLinesSpanned(self, source: str, expected_lines: List[Tuple[int, int]]) -> None:
        if False:
            return 10
        self.assertEqual(list(_line_ranges_spanned_by_format_strings(textwrap.dedent(source)).values()), expected_lines)

    def test_lines_spanned_by_format_strings(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertLinesSpanned('\n            def foo() -> None:\n                f"""\n                foo\n                {1 + "hello"}\n                bar\n                """\n\n                f"""\n                bar\n                """\n            ', [(3, 7), (9, 11)])
        self.assertLinesSpanned('\n            def foo() -> None:\n                f"{1 + "hello"}"\n            ', [(3, 3)])
        self.assertLinesSpanned('\n            def cannot_parse()\n            ', [])

    def test_map_line_to_start_of_range(self) -> None:
        if False:
            while True:
                i = 10
        self.assertEqual(_map_line_to_start_of_range([(3, 3), (3, 5), (9, 13)]), {3: 3, 4: 3, 5: 3, 9: 9, 10: 9, 11: 9, 12: 9, 13: 9})
        self.assertEqual(_map_line_to_start_of_range([]), {})
        self.assertEqual(_map_line_to_start_of_range([(3, 5), (4, 6)]), {3: 3, 4: 3, 5: 3, 6: 4})

    def test_relocate_errors(self) -> None:
        if False:
            return 10
        errors = {1: [{'code': '1', 'description': 'description'}, {'code': '2', 'description': 'description'}], 2: [{'code': '3', 'description': 'description'}, {'code': '4', 'description': 'description'}], 3: [{'code': '5', 'description': 'description'}, {'code': '6', 'description': 'description'}]}
        self.assertEqual(_relocate_errors(errors, {}), errors)
        self.assertEqual(_relocate_errors(errors, {2: 1, 3: 1}), {1: [{'code': '1', 'description': 'description'}, {'code': '2', 'description': 'description'}, {'code': '3', 'description': 'description'}, {'code': '4', 'description': 'description'}, {'code': '5', 'description': 'description'}, {'code': '6', 'description': 'description'}]})
        self.assertEqual(_relocate_errors(errors, {1: 1, 2: 2, 3: 2}), {1: [{'code': '1', 'description': 'description'}, {'code': '2', 'description': 'description'}], 2: [{'code': '3', 'description': 'description'}, {'code': '4', 'description': 'description'}, {'code': '5', 'description': 'description'}, {'code': '6', 'description': 'description'}]})