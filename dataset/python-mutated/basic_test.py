import sys
import unittest
from typing import Any, Callable, Iterable, Iterator, List, TypeVar

class BasicTestCase(unittest.TestCase):

    def test_parameter_specification(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        try:
            from .. import ParameterSpecification
            TParams = ParameterSpecification('TParams')
            TReturn = TypeVar('T')

            def listify(f: Callable[TParams, TReturn]) -> Callable[TParams, List[TReturn]]:
                if False:
                    return 10

                def wrapped(*args: TParams.args, **kwargs: TParams.kwargs):
                    if False:
                        for i in range(10):
                            print('nop')
                    return [f(*args, **kwargs)]
                return wrapped

            def foo():
                if False:
                    return 10
                return 9
            listify(foo)
        except Exception:
            self.fail('ParameterSpecification missing or broken')

    def test_typevar_tuple_variadics(self) -> None:
        if False:
            print('Hello World!')
        try:
            from .. import TypeVarTuple
            from ..type_variable_operators import Map
            TReturn = TypeVar('T')
            Ts = TypeVarTuple('Ts')

            def better_map(func: Callable[[Ts], TReturn], *args: Map[Iterable, Ts]) -> Iterator[TReturn]:
                if False:
                    return 10
                return map(func, *args)
        except Exception:
            self.fail('TypeVarTuple missing or broken')

    def test_none_throws(self) -> None:
        if False:
            i = 10
            return i + 15
        try:
            from .. import none_throws
            none_throws(0)
            none_throws(0, 'custom message')
        except Exception:
            self.fail('none_throws missing or broken')

    def test_safe_cast(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        try:
            from .. import safe_cast
            safe_cast(float, 1)
            safe_cast(1, float)
            safe_cast(Any, 'string')
        except Exception:
            self.fail('safe_cast should not have runtime implications')

    def test_generic__metaclass_conflict(self) -> None:
        if False:
            print('Hello World!')

        def try_generic_child_class() -> None:
            if False:
                return 10
            from abc import ABC, abstractmethod
            from typing import TypeVar
            from .. import Generic
            T1 = TypeVar('T1')
            T2 = TypeVar('T2')

            class Base(ABC):

                @abstractmethod
                def some_method(self) -> None:
                    if False:
                        return 10
                    ...

            class Child(Base, Generic[T1, T2]):

                def some_method(self) -> None:
                    if False:
                        while True:
                            i = 10
                    ...
        try:
            if sys.version_info >= (3, 7):
                try_generic_child_class()
            else:
                with self.assertRaises(TypeError):
                    try_generic_child_class()
        except Exception as exception:
            self.fail(f'Generic/GenericMeta/Concatenate missing or broken: Got exception `{exception}`')

    def test_variadic_tuple(self) -> None:
        if False:
            while True:
                i = 10
        try:
            from .. import TypeVarTuple, Unpack
            T = TypeVar('T')
            Ts = TypeVarTuple('Ts')

            def apply(f: Callable[[Unpack[Ts]], T], *args: Unpack[Ts]) -> T:
                if False:
                    print('Hello World!')
                return f(*args)
        except Exception:
            self.fail('Variadic tuples missing or broken')

    def test_json(self) -> None:
        if False:
            return 10
        try:
            from .. import JSON
        except Exception:
            self.fail('JSON missing or broken')

        def test_json(x: JSON) -> None:
            if False:
                print('Hello World!')
            try:
                y = x + 1
            except TypeError:
                pass
            if isinstance(x, int):
                y = x + 1
            elif isinstance(x, float):
                y = x + 1.1
            elif isinstance(x, bool):
                y = x or True
            elif isinstance(x, str):
                y = x + 'hello'
            elif isinstance(x, list):
                y = x + [4]
            elif isinstance(x, dict):
                x['key'] = 'value'
        test_json(3)
        test_json(3.5)
        test_json('test_string')
        test_json({'test': 'dict'})
        test_json(['test_list'])

    def test_readonly(self) -> None:
        if False:
            i = 10
            return i + 15
        try:
            from .. import ReadOnly

            def expect_mutable(x: int) -> ReadOnly[int]:
                if False:
                    while True:
                        i = 10
                y: ReadOnly[int] = x
                return y
        except Exception:
            self.fail('ReadOnly type is missing or broken')

    def test_override(self):
        if False:
            while True:
                i = 10
        from .. import override

        class Base:

            def normal_method(self) -> int:
                if False:
                    return 10
                ...

            @staticmethod
            def static_method_good_order() -> int:
                if False:
                    i = 10
                    return i + 15
                ...

            @staticmethod
            def static_method_bad_order() -> int:
                if False:
                    while True:
                        i = 10
                ...

            @staticmethod
            def decorator_with_slots() -> int:
                if False:
                    i = 10
                    return i + 15
                ...

        class Derived(Base):

            @override
            def normal_method(self) -> int:
                if False:
                    for i in range(10):
                        print('nop')
                return 42

            @staticmethod
            @override
            def static_method_good_order() -> int:
                if False:
                    while True:
                        i = 10
                return 42

            @override
            @staticmethod
            def static_method_bad_order() -> int:
                if False:
                    for i in range(10):
                        print('nop')
                return 42
        instance = Derived()
        self.assertEqual(instance.normal_method(), 42)
        self.assertIs(True, instance.normal_method.__override__)
        self.assertEqual(Derived.static_method_good_order(), 42)
        self.assertIs(True, Derived.static_method_good_order.__override__)
        self.assertEqual(Derived.static_method_bad_order(), 42)
        self.assertIs(False, hasattr(Derived.static_method_bad_order, '__override__'))
if __name__ == '__main__':
    unittest.main()