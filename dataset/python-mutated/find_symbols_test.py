import sys
import textwrap
from typing import List, Optional
import testslide
from ...language_server import protocol as lsp
from .. import find_symbols

def make_document_symbol(name: str, detail: str, kind: lsp.SymbolKind, range: lsp.LspRange, children: Optional[List[lsp.DocumentSymbolsResponse]]=None) -> lsp.DocumentSymbolsResponse:
    if False:
        return 10
    return lsp.DocumentSymbolsResponse(name=name, detail=detail, kind=kind, range=range, selection_range=range, children=children if children else [])
if (sys.version_info.major, sys.version_info.minor) >= (3, 8):

    class FindSymbolTests(testslide.TestCase):

        def assert_collected_symbols(self, source: str, expected_symbols: List[lsp.DocumentSymbolsResponse]) -> None:
            if False:
                print('Hello World!')
            self.maxDiff = None
            self.assertListEqual(find_symbols.parse_source_and_collect_symbols(textwrap.dedent(source)), expected_symbols)

        def test_parse_source_and_collect_symbols_function(self) -> None:
            if False:
                i = 10
                return i + 15
            self.assert_collected_symbols('\n                def foo(x):\n                    pass\n                ', [make_document_symbol(name='foo', detail='', kind=lsp.SymbolKind.FUNCTION, range=lsp.LspRange(start=lsp.LspPosition(line=1, character=0), end=lsp.LspPosition(line=2, character=len('    pass'))))])

        def test_parse_source_and_collect_symbols_function_with_variable_reassignment(self) -> None:
            if False:
                print('Hello World!')
            self.assert_collected_symbols('\n                def foo(x):\n                    x = 3\n                    [a, b, *c] = [1, 2, 3, 4]\n                    (a, b) = (1, 2)\n                    a[0] = 5\n                    pass\n                ', [make_document_symbol(name='foo', detail='', kind=lsp.SymbolKind.FUNCTION, range=lsp.LspRange(start=lsp.LspPosition(line=1, character=0), end=lsp.LspPosition(line=6, character=len('    pass'))), children=[make_document_symbol(name='x', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=2, character=4), end=lsp.LspPosition(line=2, character=5))), make_document_symbol(name='a', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=3, character=5), end=lsp.LspPosition(line=3, character=6))), make_document_symbol(name='b', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=3, character=8), end=lsp.LspPosition(line=3, character=9))), make_document_symbol(name='c', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=3, character=12), end=lsp.LspPosition(line=3, character=13))), make_document_symbol(name='a', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=4, character=5), end=lsp.LspPosition(line=4, character=6))), make_document_symbol(name='b', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=4, character=8), end=lsp.LspPosition(line=4, character=9))), make_document_symbol(name='a', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=5, character=4), end=lsp.LspPosition(line=5, character=5)))])])

        def test_parse_source_and_collect_symbols_multiple_functions(self) -> None:
            if False:
                while True:
                    i = 10
            self.assert_collected_symbols('\n                def foo(x):\n                    return x\n                def bar(y):\n                    return y\n                bar(None)\n                ', [make_document_symbol(name='foo', detail='', kind=lsp.SymbolKind.FUNCTION, range=lsp.LspRange(start=lsp.LspPosition(line=1, character=0), end=lsp.LspPosition(line=2, character=len('    return x')))), make_document_symbol(name='bar', detail='', kind=lsp.SymbolKind.FUNCTION, range=lsp.LspRange(start=lsp.LspPosition(line=3, character=0), end=lsp.LspPosition(line=4, character=len('    return y'))))])

        def test_parse_source_and_collect_symbols_annotated_atttribute(self) -> None:
            if False:
                while True:
                    i = 10
            self.assert_collected_symbols('\n                class foo:\n                    x:int = 1\n                ', [make_document_symbol(name='foo', detail='', kind=lsp.SymbolKind.CLASS, range=lsp.LspRange(start=lsp.LspPosition(line=1, character=0), end=lsp.LspPosition(line=2, character=len('    x:int = 1'))), children=[make_document_symbol(name='x', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=2, character=4), end=lsp.LspPosition(line=2, character=5)))])])

        def test_parse_source_and_collect_symbols_multiple_classes(self) -> None:
            if False:
                return 10
            self.assert_collected_symbols('\n                class foo:\n                    x = 1\n                class bar:\n                    y = 2\n                ', [make_document_symbol(name='foo', detail='', kind=lsp.SymbolKind.CLASS, range=lsp.LspRange(start=lsp.LspPosition(line=1, character=0), end=lsp.LspPosition(line=2, character=len('    x = 1'))), children=[make_document_symbol(name='x', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=2, character=4), end=lsp.LspPosition(line=2, character=5)))]), make_document_symbol(name='bar', detail='', kind=lsp.SymbolKind.CLASS, range=lsp.LspRange(start=lsp.LspPosition(line=3, character=0), end=lsp.LspPosition(line=4, character=len('    y = 2'))), children=[make_document_symbol(name='y', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=4, character=4), end=lsp.LspPosition(line=4, character=5)))])])

        def test_parse_source_and_collect_symbols_noniterable_assignment_lhs(self) -> None:
            if False:
                print('Hello World!')
            self.assert_collected_symbols('\n                class foo:\n                    w = 3\n\n                class bar:\n                    a = foo()\n                    b = ["no"]\n                    c[0] = "yes"\n                ', [make_document_symbol(name='foo', detail='', kind=lsp.SymbolKind.CLASS, range=lsp.LspRange(start=lsp.LspPosition(line=1, character=0), end=lsp.LspPosition(line=2, character=len('class foo'))), children=[make_document_symbol(name='w', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=2, character=4), end=lsp.LspPosition(line=2, character=5)))]), make_document_symbol(name='bar', detail='', kind=lsp.SymbolKind.CLASS, range=lsp.LspRange(start=lsp.LspPosition(line=4, character=0), end=lsp.LspPosition(line=7, character=len('    c[0] = "yes"'))), children=[make_document_symbol(name='a', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=5, character=4), end=lsp.LspPosition(line=5, character=5))), make_document_symbol(name='b', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=6, character=4), end=lsp.LspPosition(line=6, character=5))), make_document_symbol(name='c', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=7, character=4), end=lsp.LspPosition(line=7, character=5)))])])

        def test_parse_source_and_collect_symbols_nested_assignment(self) -> None:
            if False:
                i = 10
                return i + 15
            self.assert_collected_symbols('\n                class inner:\n                    c = 3\n\n                class middle:\n                    b = inner()\n                    b.c = 5\n\n                class outer:\n                    a = middle()\n                    a.b.c = 4\n               ', [make_document_symbol(name='inner', detail='', kind=lsp.SymbolKind.CLASS, range=lsp.LspRange(start=lsp.LspPosition(line=1, character=0), end=lsp.LspPosition(line=2, character=len('    c = 3'))), children=[make_document_symbol(name='c', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=2, character=4), end=lsp.LspPosition(line=2, character=5)))]), make_document_symbol(name='middle', detail='', kind=lsp.SymbolKind.CLASS, range=lsp.LspRange(start=lsp.LspPosition(line=4, character=0), end=lsp.LspPosition(line=6, character=len('    b.c = 4'))), children=[make_document_symbol(name='b', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=5, character=4), end=lsp.LspPosition(line=5, character=5))), make_document_symbol(name='b', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=6, character=4), end=lsp.LspPosition(line=6, character=5)))]), make_document_symbol(name='outer', detail='', kind=lsp.SymbolKind.CLASS, range=lsp.LspRange(start=lsp.LspPosition(line=8, character=0), end=lsp.LspPosition(line=10, character=len('    a.b.c = 5'))), children=[make_document_symbol(name='a', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=9, character=4), end=lsp.LspPosition(line=9, character=5))), make_document_symbol(name='a', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=10, character=4), end=lsp.LspPosition(line=10, character=5)))])])

        def test_parse_source_and_collect_symbols_list_tuple_starred_assignment_lhs(self) -> None:
            if False:
                print('Hello World!')
            self.assert_collected_symbols('\n                class foo:\n                    w = 3\n\n                class bar:\n                    a, b = (1, 2)\n                    [c, d] = [1, 2]\n                    [e, *f] = (1, 2, 3)\n                    [[g, h, [i]], [j, [k, [l]]]] = [[5, 6, [7]], [8, [9, [10]]]]\n                ', [make_document_symbol(name='foo', detail='', kind=lsp.SymbolKind.CLASS, range=lsp.LspRange(start=lsp.LspPosition(line=1, character=0), end=lsp.LspPosition(line=2, character=len('class foo'))), children=[make_document_symbol(name='w', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=2, character=4), end=lsp.LspPosition(line=2, character=5)))]), make_document_symbol(name='bar', detail='', kind=lsp.SymbolKind.CLASS, range=lsp.LspRange(start=lsp.LspPosition(line=4, character=0), end=lsp.LspPosition(line=8, character=len('    [[g, h, [j]], [j, [k, [l]]]] = [[5, 6, [7]], [8, [9, [10]]]]'))), children=[make_document_symbol(name='a', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=5, character=4), end=lsp.LspPosition(line=5, character=5))), make_document_symbol(name='b', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=5, character=7), end=lsp.LspPosition(line=5, character=8))), make_document_symbol(name='c', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=6, character=5), end=lsp.LspPosition(line=6, character=6))), make_document_symbol(name='d', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=6, character=8), end=lsp.LspPosition(line=6, character=9))), make_document_symbol(name='e', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=7, character=5), end=lsp.LspPosition(line=7, character=6))), make_document_symbol(name='f', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=7, character=9), end=lsp.LspPosition(line=7, character=10))), make_document_symbol(name='g', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=8, character=6), end=lsp.LspPosition(line=8, character=7))), make_document_symbol(name='h', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=8, character=9), end=lsp.LspPosition(line=8, character=10))), make_document_symbol(name='i', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=8, character=13), end=lsp.LspPosition(line=8, character=14))), make_document_symbol(name='j', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=8, character=19), end=lsp.LspPosition(line=8, character=20))), make_document_symbol(name='k', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=8, character=23), end=lsp.LspPosition(line=8, character=24))), make_document_symbol(name='l', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=8, character=27), end=lsp.LspPosition(line=8, character=28)))])])

        def test_parse_source_and_collect_symbols_multiple_classes_and_class_attributes(self) -> None:
            if False:
                while True:
                    i = 10
            self.assert_collected_symbols('\n                class foo:\n                    x = z = 1\n                class bar:\n                    y = w = 2\n                ', [make_document_symbol(name='foo', detail='', kind=lsp.SymbolKind.CLASS, range=lsp.LspRange(start=lsp.LspPosition(line=1, character=0), end=lsp.LspPosition(line=2, character=len('    x = z = 1'))), children=[make_document_symbol(name='x', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=2, character=4), end=lsp.LspPosition(line=2, character=5))), make_document_symbol(name='z', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=2, character=8), end=lsp.LspPosition(line=2, character=9)))]), make_document_symbol(name='bar', detail='', kind=lsp.SymbolKind.CLASS, range=lsp.LspRange(start=lsp.LspPosition(line=3, character=0), end=lsp.LspPosition(line=4, character=len('    y = w = 2'))), children=[make_document_symbol(name='y', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=4, character=4), end=lsp.LspPosition(line=4, character=5))), make_document_symbol(name='w', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=4, character=8), end=lsp.LspPosition(line=4, character=9)))])])

        def test_parse_source_and_collect_symbols_method_with_assignment(self) -> None:
            if False:
                print('Hello World!')
            self.maxDiff = None
            self.assert_collected_symbols('\n                class foo:\n                    def bar(self):\n                        w = 2\n                        return self\n                ', [make_document_symbol(name='foo', detail='', kind=lsp.SymbolKind.CLASS, range=lsp.LspRange(start=lsp.LspPosition(line=1, character=0), end=lsp.LspPosition(line=4, character=len('        return self'))), children=[make_document_symbol(name='bar', detail='', kind=lsp.SymbolKind.FUNCTION, range=lsp.LspRange(start=lsp.LspPosition(line=2, character=4), end=lsp.LspPosition(line=4, character=len('        return self'))), children=[make_document_symbol(name='w', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=3, character=8), end=lsp.LspPosition(line=3, character=9)))])])])

        def test_parse_source_and_collect_symbols_method(self) -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.maxDiff = None
            self.assert_collected_symbols('\n                class foo:\n                    def bar(self):\n                        return self\n                ', [make_document_symbol(name='foo', detail='', kind=lsp.SymbolKind.CLASS, range=lsp.LspRange(start=lsp.LspPosition(line=1, character=0), end=lsp.LspPosition(line=3, character=len('        return self'))), children=[make_document_symbol(name='bar', detail='', kind=lsp.SymbolKind.FUNCTION, range=lsp.LspRange(start=lsp.LspPosition(line=2, character=4), end=lsp.LspPosition(line=3, character=len('        return self'))))])])

        def test_parse_source_and_collect_symbols_nested_classes(self) -> None:
            if False:
                print('Hello World!')
            self.assert_collected_symbols('\n                class foo:\n                    class bar:\n                        def foobar(self):\n                            return self\n                ', [make_document_symbol(name='foo', detail='', kind=lsp.SymbolKind.CLASS, range=lsp.LspRange(start=lsp.LspPosition(line=1, character=0), end=lsp.LspPosition(line=4, character=len('            return self'))), children=[make_document_symbol(name='bar', detail='', kind=lsp.SymbolKind.CLASS, range=lsp.LspRange(start=lsp.LspPosition(line=2, character=4), end=lsp.LspPosition(line=4, character=len('            return self'))), children=[make_document_symbol(name='foobar', detail='', kind=lsp.SymbolKind.FUNCTION, range=lsp.LspRange(start=lsp.LspPosition(line=3, character=8), end=lsp.LspPosition(line=4, character=len('            return self'))))])])])

        def test_parse_source_and_collect_symbols_nested_funcs(self) -> None:
            if False:
                i = 10
                return i + 15
            self.assert_collected_symbols('\n                def foo(x):\n                    def bar(y):\n                        def foobar(xy):\n                            return x * y * xy\n                        foobar(y)\n                    return bar(x)\n                ', [make_document_symbol(name='foo', detail='', kind=lsp.SymbolKind.FUNCTION, range=lsp.LspRange(start=lsp.LspPosition(line=1, character=0), end=lsp.LspPosition(line=6, character=len('    return bar(x)'))), children=[make_document_symbol(name='bar', detail='', kind=lsp.SymbolKind.FUNCTION, range=lsp.LspRange(start=lsp.LspPosition(line=2, character=4), end=lsp.LspPosition(line=5, character=len('        foobar(y)'))), children=[make_document_symbol(name='foobar', detail='', kind=lsp.SymbolKind.FUNCTION, range=lsp.LspRange(start=lsp.LspPosition(line=3, character=8), end=lsp.LspPosition(line=4, character=len('            return x * y * xy'))))])])])

        def test_parse_source_and_collect_async_funcs(self) -> None:
            if False:
                i = 10
                return i + 15
            self.assert_collected_symbols('\n                async def  foo(x):\n                    await x\n                ', [make_document_symbol(name='foo', detail='', kind=lsp.SymbolKind.FUNCTION, range=lsp.LspRange(start=lsp.LspPosition(line=1, character=0), end=lsp.LspPosition(line=2, character=len('    await x'))))])

        def test_parse_source_and_collect_symbols_invalid_syntax(self) -> None:
            if False:
                print('Hello World!')
            self.assertRaises(find_symbols.UnparseableError, find_symbols.parse_source_and_collect_symbols, 'thisIsNotValidPython x = x')

        def test_parse_source_and_collect_symbols_multiple_calls(self) -> None:
            if False:
                for i in range(10):
                    print('nop')
            for _ in range(2):
                self.assert_collected_symbols('\n                            def foo(x):\n                                pass\n                            ', [make_document_symbol(name='foo', detail='', kind=lsp.SymbolKind.FUNCTION, range=lsp.LspRange(start=lsp.LspPosition(line=1, character=0), end=lsp.LspPosition(line=2, character=len('    pass'))))])

        def test_parse_source_and_collect_symbols_enums_from_import(self) -> None:
            if False:
                return 10
            self.assert_collected_symbols('\n                from enum import Enum\n\n                class Animal(Enum):\n                    cat = 1\n                    dog = 2\n                    lion = 3\n\n                ', [make_document_symbol(name='Animal', detail='', kind=lsp.SymbolKind.CLASS, range=lsp.LspRange(start=lsp.LspPosition(line=3, character=0), end=lsp.LspPosition(line=6, character=len('    lion = 3'))), children=[make_document_symbol(name='cat', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=4, character=4), end=lsp.LspPosition(line=4, character=7))), make_document_symbol(name='dog', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=5, character=4), end=lsp.LspPosition(line=5, character=7))), make_document_symbol(name='lion', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=6, character=4), end=lsp.LspPosition(line=6, character=8)))])])

        def test_parse_source_and_collect_symbols_enums(self) -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.assert_collected_symbols('\n                import enum\n\n                class Animal(enum.Enum):\n                    cat = 1\n                    dog = 2\n                    lion = 3\n\n                ', [make_document_symbol(name='Animal', detail='', kind=lsp.SymbolKind.CLASS, range=lsp.LspRange(start=lsp.LspPosition(line=3, character=0), end=lsp.LspPosition(line=6, character=len('    lion = 3'))), children=[make_document_symbol(name='cat', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=4, character=4), end=lsp.LspPosition(line=4, character=7))), make_document_symbol(name='dog', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=5, character=4), end=lsp.LspPosition(line=5, character=7))), make_document_symbol(name='lion', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=6, character=4), end=lsp.LspPosition(line=6, character=8)))])])

        def test_parse_source_and_collect_symbols_int_enums_from_import(self) -> None:
            if False:
                i = 10
                return i + 15
            self.assert_collected_symbols('\n                from enum import IntEnum\n\n                class Animal(IntEnum):\n                    cat = 1\n                    dog = 2\n                    lion = 3\n\n                ', [make_document_symbol(name='Animal', detail='', kind=lsp.SymbolKind.CLASS, range=lsp.LspRange(start=lsp.LspPosition(line=3, character=0), end=lsp.LspPosition(line=6, character=len('    lion = 3'))), children=[make_document_symbol(name='cat', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=4, character=4), end=lsp.LspPosition(line=4, character=7))), make_document_symbol(name='dog', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=5, character=4), end=lsp.LspPosition(line=5, character=7))), make_document_symbol(name='lion', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=6, character=4), end=lsp.LspPosition(line=6, character=8)))])])

        def test_parse_source_and_collect_symbols_int_enums(self) -> None:
            if False:
                return 10
            self.assert_collected_symbols('\n                import enum\n\n                class Animal(enum.IntEnum):\n                    cat = 1\n                    dog = 2\n                    lion = 3\n\n                ', [make_document_symbol(name='Animal', detail='', kind=lsp.SymbolKind.CLASS, range=lsp.LspRange(start=lsp.LspPosition(line=3, character=0), end=lsp.LspPosition(line=6, character=len('    lion = 3'))), children=[make_document_symbol(name='cat', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=4, character=4), end=lsp.LspPosition(line=4, character=7))), make_document_symbol(name='dog', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=5, character=4), end=lsp.LspPosition(line=5, character=7))), make_document_symbol(name='lion', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=6, character=4), end=lsp.LspPosition(line=6, character=8)))])])

        def test_parse_source_and_collect_symbols_typevar(self) -> None:
            if False:
                print('Hello World!')
            self.assert_collected_symbols('\n                from typing import TypeVar, Optional\n\n                T = TypeVar("T")\n\n                def get_first_item(items: List[T]) -> Optional[T]:\n                    return items[0] if len(items) > 0 else None\n\n                ', [make_document_symbol(name='T', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=3, character=0), end=lsp.LspPosition(line=3, character=1))), make_document_symbol(name='get_first_item', detail='', kind=lsp.SymbolKind.FUNCTION, range=lsp.LspRange(start=lsp.LspPosition(line=5, character=0), end=lsp.LspPosition(line=6, character=len('    return items[0] if len(items) > 0 else None'))))])

        def test_parse_source_and_collect_symbols_global_var(self) -> None:
            if False:
                while True:
                    i = 10
            self.assert_collected_symbols('\n                cost = 5\n\n                def get_total_cost(num_items: int) -> int:\n                    return num_items * cost\n\n                ', [make_document_symbol(name='cost', detail='', kind=lsp.SymbolKind.VARIABLE, range=lsp.LspRange(start=lsp.LspPosition(line=1, character=0), end=lsp.LspPosition(line=1, character=4))), make_document_symbol(name='get_total_cost', detail='', kind=lsp.SymbolKind.FUNCTION, range=lsp.LspRange(start=lsp.LspPosition(line=3, character=0), end=lsp.LspPosition(line=4, character=len('    return num_items * cost'))))])