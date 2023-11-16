"""Python 3 tests for yapf.reformatter."""
import sys
import textwrap
import unittest
from yapf.yapflib import reformatter
from yapf.yapflib import style
from yapftests import yapf_test_helper

class TestsForPython3Code(yapf_test_helper.YAPFTest):
    """Test a few constructs that are new Python 3 syntax."""

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        style.SetGlobalStyle(style.CreatePEP8Style())

    def testTypedNames(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        def x(aaaaaaaaaaaaaaa:int,bbbbbbbbbbbbbbbb:str,ccccccccccccccc:dict,eeeeeeeeeeeeee:set={1, 2, 3})->bool:\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        def x(aaaaaaaaaaaaaaa: int,\n              bbbbbbbbbbbbbbbb: str,\n              ccccccccccccccc: dict,\n              eeeeeeeeeeeeee: set = {1, 2, 3}) -> bool:\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testTypedNameWithLongNamedArg(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        def func(arg=long_function_call_that_pushes_the_line_over_eighty_characters()) -> ReturnType:\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        def func(arg=long_function_call_that_pushes_the_line_over_eighty_characters()\n                 ) -> ReturnType:\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testKeywordOnlyArgSpecifier(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        def foo(a, *, kw):\n          return a+kw\n    ')
        expected_formatted_code = textwrap.dedent('        def foo(a, *, kw):\n            return a + kw\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testAnnotations(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('        def foo(a: list, b: "bar") -> dict:\n          return a+b\n    ')
        expected_formatted_code = textwrap.dedent('        def foo(a: list, b: "bar") -> dict:\n            return a + b\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testExecAsNonKeyword(self):
        if False:
            return 10
        unformatted_code = 'methods.exec( sys.modules[name])\n'
        expected_formatted_code = 'methods.exec(sys.modules[name])\n'
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testAsyncFunctions(self):
        if False:
            return 10
        code = textwrap.dedent('        import asyncio\n        import time\n\n\n        @print_args\n        async def slow_operation():\n            await asyncio.sleep(1)\n            # print("Slow operation {} complete".format(n))\n\n\n        async def main():\n            start = time.time()\n            if (await get_html()):\n                pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testNoSpacesAroundPowerOperator(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        a**b\n    ')
        expected_formatted_code = textwrap.dedent('        a ** b\n    ')
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: pep8, SPACES_AROUND_POWER_OPERATOR: True}'))
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreatePEP8Style())

    def testSpacesAroundDefaultOrNamedAssign(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        f(a=5)\n    ')
        expected_formatted_code = textwrap.dedent('        f(a = 5)\n    ')
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: pep8, SPACES_AROUND_DEFAULT_OR_NAMED_ASSIGN: True}'))
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreatePEP8Style())

    def testTypeHint(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent("        def foo(x: int=42):\n            pass\n\n\n        def foo2(x: 'int' =42):\n            pass\n    ")
        expected_formatted_code = textwrap.dedent("        def foo(x: int = 42):\n            pass\n\n\n        def foo2(x: 'int' = 42):\n            pass\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testMatrixMultiplication(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        a=b@c\n    ')
        expected_formatted_code = textwrap.dedent('        a = b @ c\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testNoneKeyword(self):
        if False:
            return 10
        code = textwrap.dedent('        None.__ne__()\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testAsyncWithPrecedingComment(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent('        import asyncio\n\n        # Comment\n        async def bar():\n            pass\n\n        async def foo():\n            pass\n    ')
        expected_formatted_code = textwrap.dedent('        import asyncio\n\n\n        # Comment\n        async def bar():\n            pass\n\n\n        async def foo():\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testAsyncFunctionsNested(self):
        if False:
            return 10
        code = textwrap.dedent('        async def outer():\n\n            async def inner():\n                pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testKeepTypesIntact(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        def _ReduceAbstractContainers(\n            self, *args: Optional[automation_converter.PyiCollectionAbc]) -> List[\n                automation_converter.PyiCollectionAbc]:\n            pass\n    ')
        expected_formatted_code = textwrap.dedent('        def _ReduceAbstractContainers(\n            self, *args: Optional[automation_converter.PyiCollectionAbc]\n        ) -> List[automation_converter.PyiCollectionAbc]:\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testContinuationIndentWithAsync(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        async def start_websocket():\n            async with session.ws_connect(\n                r"ws://a_really_long_long_long_long_long_long_url") as ws:\n                pass\n    ')
        expected_formatted_code = textwrap.dedent('        async def start_websocket():\n            async with session.ws_connect(\n                    r"ws://a_really_long_long_long_long_long_long_url") as ws:\n                pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testSplittingArguments(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent("        async def open_file(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):\n            pass\n\n        async def run_sync_in_worker_thread(sync_fn, *args, cancellable=False, limiter=None):\n            pass\n\n        def open_file(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):\n            pass\n\n        def run_sync_in_worker_thread(sync_fn, *args, cancellable=False, limiter=None):\n            pass\n    ")
        expected_formatted_code = textwrap.dedent("        async def open_file(\n            file,\n            mode='r',\n            buffering=-1,\n            encoding=None,\n            errors=None,\n            newline=None,\n            closefd=True,\n            opener=None\n        ):\n            pass\n\n\n        async def run_sync_in_worker_thread(\n            sync_fn, *args, cancellable=False, limiter=None\n        ):\n            pass\n\n\n        def open_file(\n            file,\n            mode='r',\n            buffering=-1,\n            encoding=None,\n            errors=None,\n            newline=None,\n            closefd=True,\n            opener=None\n        ):\n            pass\n\n\n        def run_sync_in_worker_thread(sync_fn, *args, cancellable=False, limiter=None):\n            pass\n    ")
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: pep8, dedent_closing_brackets: true, coalesce_brackets: false, space_between_ending_comma_and_closing_bracket: false, split_arguments_when_comma_terminated: true, split_before_first_argument: true}'))
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreatePEP8Style())

    def testDictUnpacking(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent("        class Foo:\n            def foo(self):\n                foofoofoofoofoofoofoofoo('foofoofoofoofoo', {\n\n                    'foo': 'foo',\n\n                    **foofoofoo\n                })\n    ")
        expected_formatted_code = textwrap.dedent("        class Foo:\n\n            def foo(self):\n                foofoofoofoofoofoofoofoo('foofoofoofoofoo', {\n                    'foo': 'foo',\n                    **foofoofoo\n                })\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testMultilineFormatString(self):
        if False:
            for i in range(10):
                print('nop')
        code = textwrap.dedent("        # yapf: disable\n        (f'''\n          ''')\n        # yapf: enable\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testEllipses(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        def dirichlet(x12345678901234567890123456789012345678901234567890=...) -> None:\n            return\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testFunctionTypedReturnNextLine(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent('        def _GenerateStatsEntries(\n            process_id: Text,\n            timestamp: Optional[ffffffff.FFFFFFFFFFF] = None\n        ) -> Sequence[ssssssssssss.SSSSSSSSSSSSSSS]:\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testFunctionTypedReturnSameLine(self):
        if False:
            return 10
        code = textwrap.dedent('        def rrrrrrrrrrrrrrrrrrrrrr(\n                ccccccccccccccccccccccc: Tuple[Text, Text]) -> List[Tuple[Text, Text]]:\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testAsyncForElseNotIndentedInsideBody(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        async def fn():\n            async for message in websocket:\n                for i in range(10):\n                    pass\n                else:\n                    pass\n            else:\n                pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testForElseInAsyncNotMixedWithAsyncFor(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent('        async def fn():\n            for i in range(10):\n                pass\n            else:\n                pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testParameterListIndentationConflicts(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent("        def raw_message(  # pylint: disable=too-many-arguments\n                    self, text, user_id=1000, chat_type='private', forward_date=None, forward_from=None):\n                pass\n    ")
        expected_formatted_code = textwrap.dedent("        def raw_message(  # pylint: disable=too-many-arguments\n                self,\n                text,\n                user_id=1000,\n                chat_type='private',\n                forward_date=None,\n                forward_from=None):\n            pass\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testTypeHintedYieldExpression(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('       def my_coroutine():\n           x: int = yield\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testSyntaxMatch(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('        a=3\n        b=0\n        match a :\n            case 0 :\n                b=1\n            case _\t:\n                b=2\n    ')
        expected_formatted_code = textwrap.dedent('        a = 3\n        b = 0\n        match a:\n            case 0:\n                b = 1\n            case _:\n                b = 2\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testParenthsizedContextManager(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent('        def test_copy_dimension(self):\n            with (Dataset() as target_ds,\n                  Dataset() as source_ds):\n                do_something\n    ')
        expected_formatted_code = textwrap.dedent('        def test_copy_dimension(self):\n            with (Dataset() as target_ds, Dataset() as source_ds):\n                do_something\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testUnpackedTuple(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        def a():\n          t = (2,3)\n          for i in range(5):\n            yield i,*t\n    ')
        expected_formatted_code = textwrap.dedent('        def a():\n            t = (2, 3)\n            for i in range(5):\n                yield i, *t\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testTypedTuple(self):
        if False:
            for i in range(10):
                print('nop')
        code = textwrap.dedent('        t: tuple = 1, 2\n        args = tuple(x for x in [2], )\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testWalrusOperator(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent('        import os\n        a=[1,2,3,4]\n        if (n:=len(a))>2:\n            print()\n    ')
        expected_formatted_code = textwrap.dedent('        import os\n\n        a = [1, 2, 3, 4]\n        if (n := len(a)) > 2:\n            print()\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testCondAssign(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        def json(self) -> JSONTask:\n                result: JSONTask = {\n                    "id": self.id,\n                    "text": self.text,\n                    "status": self.status,\n                    "last_mod": self.last_mod_time\n                }\n                for i in "parent_id", "deadline", "reminder":\n                    if x := getattr(self , i):\n                        result[i] = x  # type: ignore\n                return result\n    ')
        expected_formatted_code = textwrap.dedent('        def json(self) -> JSONTask:\n            result: JSONTask = {\n                "id": self.id,\n                "text": self.text,\n                "status": self.status,\n                "last_mod": self.last_mod_time\n            }\n            for i in "parent_id", "deadline", "reminder":\n                if x := getattr(self, i):\n                    result[i] = x  # type: ignore\n            return result\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testCopyDictionary(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent("        a_dict = {'key': 'value'}\n        a_dict_copy = {**a_dict}\n        print('a_dict:', a_dict)\n        print('a_dict_copy:', a_dict_copy)\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))
if __name__ == '__main__':
    unittest.main()