import os
import textwrap
import unittest
from typing import Any, Callable, Dict, IO, Iterable, Optional, Set
from unittest.mock import call, mock_open, patch
from ..get_globals import __name__ as get_globals_name, GlobalModelGenerator

def _open_implementation(path_to_content: Dict[str, str]) -> Callable[[str, str], Any]:
    if False:
        i = 10
        return i + 15

    def _nested_open_implementation(path: str, mode: str) -> IO[Any]:
        if False:
            print('Hello World!')
        if path in path_to_content:
            return mock_open(read_data=path_to_content[path]).return_value
        else:
            raise FileNotFoundError(path)
    return _nested_open_implementation

class GetGlobalsTest(unittest.TestCase):

    @patch('os.path.exists', side_effect=lambda path: path in {'/root/a.py', '/root/a.pyi', '/root/b.py'} or '/stub_root' in path)
    @patch('os.path.abspath', side_effect=lambda path: path)
    @patch('os.getcwd', return_value='/root')
    def test_get_globals(self, current_working_directory: unittest.mock._patch, absolute_path: unittest.mock._patch, exists: unittest.mock._patch) -> None:
        if False:
            print('Hello World!')
        with patch(f'{get_globals_name}.GlobalModelGenerator._globals') as globals, patch('glob.glob', return_value=['/root/a.py', '/root/b.py']):
            GlobalModelGenerator(root='/root', stub_root='/stub_root').compute_models([])
            globals.assert_has_calls([call('/root', '/root/a.pyi'), call('/root', '/root/b.py')], any_order=True)
        directory_mapping = {'/root/**/*.py': ['/root/a.py', '/root/b.py'], '/stub_root/**/*.pyi': ['/stub_root/a.pyi', '/stub_root/b.pyi']}
        with patch(f'{get_globals_name}.GlobalModelGenerator._globals') as globals, patch('glob.glob', side_effect=lambda root, recursive: directory_mapping[root]):
            GlobalModelGenerator(root='/root', stub_root='/stub_root').compute_models([])
            globals.assert_has_calls([call('/root', '/root/a.pyi'), call('/root', '/root/b.py'), call('/stub_root', '/stub_root/a.pyi'), call('/stub_root', '/stub_root/b.pyi')], any_order=True)

    def assert_module_has_global_models(self, source: str, expected: Iterable[str], blacklist: Optional[Set[str]]=None) -> None:
        if False:
            return 10
        blacklist = blacklist or set()
        with patch('builtins.open') as open:
            open.side_effect = _open_implementation({'/root/module.py': textwrap.dedent(source)})
            generator = GlobalModelGenerator(root='/root', blacklisted_globals=blacklist)
            self.assertSetEqual({str(model) for model in generator._globals('/root', '/root/module.py')}, set(expected))

    @patch('builtins.open')
    def test_globals(self, open: unittest.mock._patch) -> None:
        if False:
            return 10
        self.assert_module_has_global_models('\n            A = 1\n            def function():\n              B = 2\n            if "version" is None:\n              C = 2\n            D, E = 1, 2\n            __all__ = {}\n            ', {'module.A: TaintSink[Global] = ...', 'module.D: TaintSink[Global] = ...', 'module.E: TaintSink[Global] = ...'})
        self.assert_module_has_global_models('\n            class Class:\n              F: typing.ClassVar[int] = ...\n              G: int = ...\n              class Nested:\n                H: typing.ClassVar[int] = ...\n            ', {'module.Class.__class__.F: TaintSink[Global] = ...', 'module.Class.__class__.G: TaintSink[Global] = ...'})
        self.assert_module_has_global_models('\n            Z.X = 1\n            A, B.C, D = 1, 2, 3\n            [Y, Q.W] = [1, 2]\n            ', {'module.A: TaintSink[Global] = ...', 'module.D: TaintSink[Global] = ...', 'module.Y: TaintSink[Global] = ...'})
        self.assert_module_has_global_models('\n            from collections import namedtuple\n            x = collections.namedtuple()\n            y = namedtuple()\n            ', set())
        self.assert_module_has_global_models('\n            x = a\n            y = b.c\n            ', set())
        self.assert_module_has_global_models('\n            x[1] = 123\n            y.field = 456\n            ', set())
        self.assert_module_has_global_models('\n            x: int = 1\n            y: str  # this is ignored, as it might not exist in the runtime\n            z: Any = alias_that_we_skip\n            ', {'module.x: TaintSink[Global] = ...'})
        self.assert_module_has_global_models('\n            A, B = 1\n            class Class:\n              C: typing.ClassVar[int] = ...\n              D: int = ...\n            ', expected={'module.B: TaintSink[Global] = ...', 'module.Class.__class__.D: TaintSink[Global] = ...'}, blacklist={'module.A', 'module.Class.__class__.C'})
        self.assert_module_has_global_models('\n            from dataclasses import dataclass\n            @dataclass\n            class Class:\n              C: int = ...\n              D: int = ...\n            @dataclass(frozen=True)\n            class Frozen:\n              C: int = ...\n              D: int = ...\n            ', set())
        self.assert_module_has_global_models('\n            import dataclasses\n            @dataclasses.dataclass\n            class Class:\n              C: int = ...\n              D: int = ...\n            @dataclasses.dataclass(frozen=True)\n            class Frozen:\n              C: int = ...\n              D: int = ...\n            ', set())
        self.assert_module_has_global_models('\n            class MyClass:\n              C: int = ...\n              D: int = ...\n              def __init__(self):\n                self.C = 1\n            ', {'module.MyClass.__class__.D: TaintSink[Global] = ...'})
        self.assert_module_has_global_models('\n            class MyClass:\n              C: ClassVar[int] = ...\n              def __init__(self):\n                self.C = 1\n            ', set())
        self.assert_module_has_global_models('\n            class MyClass:\n              C: ClassVar[int] = ...\n              def foo(self):\n                self.C = 1\n            ', set())
        self.assert_module_has_global_models('\n            class MyClass:\n              C: int = ...\n              def __init__(self):\n                self.C = 1\n            class SubClass(MyClass):\n              C: int = ...\n            ', {'module.SubClass.__class__.C: TaintSink[Global] = ...'})
        self.assert_module_has_global_models('\n            from typing import TypedDict\n            class MyClass(TypedDict):\n              x: int = ...\n              y: str = ...\n            ', {})
        self.assert_module_has_global_models('\n            import typing\n            class MyClass(typing.TypedDict):\n              x: int = ...\n              y: str = ...\n            ', {})
        self.assert_module_has_global_models('\n            class MyClass:\n              x = lambda x: y\n            ', {})
        self.assert_module_has_global_models('\n            class MyClass:\n              @property\n              def foo():\n                return 0\n            ', {})
        self.assert_module_has_global_models('\n            class MyClass:\n              @cached_property\n              def foo(self):\n                return 0\n            ', {'def module.MyClass.foo(self) -> TaintSink[Global, Via[cached_property]]: ...'})
        self.assert_module_has_global_models('\n            class MyClass:\n              @util.some_property_module.cached_property\n              def foo(self):\n                return 0\n            ', {'def module.MyClass.foo(self) -> TaintSink[Global, Via[cached_property]]: ...'})
        self.assert_module_has_global_models('\n            class MyClass:\n              @cached_classproperty\n              def foo(self):\n                return 0\n            ', {'def module.MyClass.foo(self) -> TaintSink[Global, Via[cached_class_property]]: ...'})