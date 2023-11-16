"""Tests for inferring protocols."""
from pytype.tests import test_base
from pytype.tests import test_utils

class ProtocolInferenceTest(test_base.BaseTest):
    """Tests for protocol implementation."""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.options.tweak(protocols=True)

    def test_multiple_signatures_with_type_parameter(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import List, TypeVar\n        T = TypeVar("T")\n        def f(x: T, y: int) -> List[T]: ...\n        def f(x: List[T], y: str) -> List[T]: ...\n      ')
            ty = self.Infer('\n        import foo\n        def f(x, y):\n          return foo.f(x, y)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Union\n        def f(x, y: Union[int, str]) -> list: ...\n      ')

    def test_unknown_single_signature(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import List, TypeVar\n        T = TypeVar("T")\n        def f(x: T, y: int) -> List[T]: ...\n        def f(x: List[T], y: str) -> List[T]: ...\n      ')
            ty = self.Infer('\n        import foo\n        def f(y):\n          return foo.f("", y)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import List\n        def f(y: int) -> List[str]: ...\n      ')

    def test_multiple_signatures_with_unknown(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(arg1: str) -> float: ...\n        def f(arg2: int) -> bool: ...\n      ')
            ty = self.Infer('\n        import foo\n        def f(x):\n          return foo.f(x)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Union\n        def f(x: Union[int, str]) -> Union[float, bool]: ...\n      ')

    def test_multiple_signatures_with_optional_arg(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(x: str) -> int: ...\n        def f(x = ...) -> float: ...\n      ')
            ty = self.Infer('\n        import foo\n        def f(x):\n          return foo.f(x)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Union\n        def f(x: str) -> Union[int, float]: ...\n      ')

    def test_multiple_signatures_with_kwarg(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(*, y: int) -> bool: ...\n        def f(y: str) -> float: ...\n      ')
            ty = self.Infer('\n        import foo\n        def f(x):\n          return foo.f(y=x)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Union\n        def f(x: Union[int, str]) -> Union[bool, float]: ...\n      ')

    def test_pow2(self):
        if False:
            return 10
        ty = self.Infer("\n      def t_testPow2(x, y):\n        # pow(int, int) returns int, or float if the exponent is negative.\n        # Hence, it's a handy function for testing UnionType returns.\n        return pow(x, y)\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      def t_testPow2(x: Union[complex, float, int], y: Union[complex, float, int]) -> Union[complex, float, int]: ...\n    ')

    @test_base.skip('Moving to protocols.')
    def test_slices(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def trim(docstring):\n        lines = docstring.splitlines()\n        for line in lines[1:]:\n          len(line)\n        return lines\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List, Union\n      def trim(docstring: Union[bytearray, str, unicode]) -> List[Union[bytearray, str, unicode], ...]: ...\n    ')

    def test_match_unknown_against_container(self):
        if False:
            return 10
        ty = self.Infer('\n      a = {1}\n      def f(x):\n        return a & x\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Set\n      a = ...  # type: Set[int]\n\n      def f(x) -> Set[int]: ...\n    ')

    def test_supports_lower(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f(x):\n        return x.lower()\n     ')
        self.assertTypesMatchPytd(ty, '\n      import protocols\n      from typing import Any\n      def f(x: protocols.SupportsLower) -> Any: ...\n    ')

    def test_container(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      def f(x, y):\n          return y in x\n     ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Container\n      def f(x: Container, y:Any) -> bool: ...\n    ')

    def test_supports_int(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      def f(x):\n        return x.__int__()\n      ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, SupportsInt\n      def f(x: SupportsInt) -> Any: ...\n    ')

    def test_supports_float(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f(x):\n          return x.__float__()\n      ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, SupportsFloat\n      def f(x: SupportsFloat) -> Any: ...\n    ')

    def test_supports_complex(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f(x):\n        return x.__complex__()\n      ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, SupportsComplex\n      def f(x: SupportsComplex) -> Any: ...\n    ')

    def test_sized(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f(x):\n        return x.__len__()\n      ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Sized\n      def f(x: Sized) -> Any: ...\n    ')

    def test_supports_abs(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f(x):\n        y = abs(x)\n        return y.__len__()\n      ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, SupportsAbs, Sized\n      def f(x: SupportsAbs[Sized]) -> Any: ...\n    ')

    @test_base.skip("doesn't match arguments correctly")
    def test_supports_round(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f(x):\n        y = x.__round__()\n      ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, SupportsRound\n      def f(x: SupportsRound) -> Any: ...\n    ')

    def test_reversible(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f(x):\n        y = x.__reversed__()\n        return y\n      ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Iterator, Reversible\n      def f(x: Reversible) -> Iterator: ...\n    ')

    def test_iterable(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f(x):\n        return x.__iter__()\n      ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Iterable, Iterator\n      def f(x: Iterable) -> Iterator: ...\n    ')

    @test_base.skip('Iterator not implemented, breaks other functionality')
    def test_iterator(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      def f(x):\n        return x.next()\n      ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Iterator\n      def f(x: Iterator) -> Any: ...\n    ')

    def test_callable(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f(x):\n        return x().lower()\n      ')
        self.assertTypesMatchPytd(ty, '\n      import protocols\n      from typing import Any, Callable\n      def f(x: Callable[..., protocols.SupportsLower]) -> Any: ...\n    ')

    @test_base.skip('Matches Mapping[int, Any] but not Sequence')
    def test_sequence(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f(x):\n        x.index(6)\n        x.count(7)\n        return x.__getitem__(5) + x[1:5]\n      ')
        self.assertTypesMatchPytd(ty, '\n      import protocols\n      from typing import Any, Sequence\n      def f(x: Sequence) -> Any: ...\n    ')

    @test_base.skip("doesn't match arguments correctly on exit")
    def test_context_manager(self):
        if False:
            return 10
        ty = self.Infer('\n      def f(x):\n        x.__enter__()\n        x.__exit__(None, None, None)\n      ')
        self.assertTypesMatchPytd(ty, '\n      import protocols\n      from typing import Any, ContextManager\n      def f(x: ContextManager) -> Any: ...\n    ')

    def test_protocol_needs_parameter(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Sized, SupportsAbs\n        def f(x: SupportsAbs[Sized]) -> None: ...\n      ')
            ty = self.Infer('\n        import foo\n        def g(y):\n          return foo.f(y)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Sized, SupportsAbs\n        def g(y: SupportsAbs[Sized]) -> None: ...\n      ')

    def test_protocol_needs_parameter_builtin(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import SupportsAbs\n        def f(x: SupportsAbs[int]) -> None: ...\n      ')
            ty = self.Infer('\n        import foo\n        def g(y):\n          return foo.f(y)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import SupportsAbs\n        def g(y: SupportsAbs[int]) -> None: ...\n      ')

    @test_base.skip('Unexpectedly assumes returned result is sequence')
    def test_mapping_abstractmethod(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      def f(x, y):\n        return x.__getitem__(y)\n      ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Mapping\n      def f(x: Mapping, y) -> Any: ...\n    ')

    def test_supports_upper(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f(x):\n        return x.upper()\n     ')
        self.assertTypesMatchPytd(ty, '\n      import protocols\n      from typing import Any\n      def f(x: protocols.SupportsUpper) -> Any: ...\n    ')

    def test_supports_startswith(self):
        if False:
            return 10
        ty = self.Infer('\n      def f(x):\n        return x.startswith("foo")\n     ')
        self.assertTypesMatchPytd(ty, '\n      import protocols\n      from typing import Any\n      def f(x: protocols.SupportsStartswith) -> Any: ...\n    ')

    def test_supports_endswith(self):
        if False:
            return 10
        ty = self.Infer('\n      def f(x):\n        return x.endswith("foo")\n     ')
        self.assertTypesMatchPytd(ty, '\n      import protocols\n      from typing import Any\n      def f(x: protocols.SupportsEndswith) -> Any: ...\n    ')

    def test_supports_lstrip(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f(x):\n        return x.lstrip()\n     ')
        self.assertTypesMatchPytd(ty, '\n      import protocols\n      from typing import Any\n      def f(x: protocols.SupportsLstrip) -> Any: ...\n    ')

    def test_supports_replace(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f(x):\n        return x.replace("foo", "bar")\n     ')
        self.assertTypesMatchPytd(ty, '\n      import protocols\n      from typing import Any\n      def f(x: protocols.SupportsReplace) -> Any: ...\n    ')

    def test_supports_encode(self):
        if False:
            return 10
        ty = self.Infer('\n      def f(x):\n        return x.encode()\n     ')
        self.assertTypesMatchPytd(ty, '\n      import protocols\n      from typing import Any\n      def f(x: protocols.SupportsEncode) -> Any: ...\n    ')

    def test_supports_decode(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f(x):\n        return x.decode()\n     ')
        self.assertTypesMatchPytd(ty, '\n      import protocols\n      from typing import Any\n      def f(x: protocols.SupportsDecode) -> Any: ...\n    ')

    def test_supports_splitlines(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f(x):\n        return x.splitlines()\n     ')
        self.assertTypesMatchPytd(ty, '\n      import protocols\n      from typing import Any\n      def f(x: protocols.SupportsSplitlines) -> Any: ...\n    ')

    def test_supports_split(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f(x):\n        return x.split()\n     ')
        self.assertTypesMatchPytd(ty, '\n      import protocols\n      from typing import Any\n      def f(x: protocols.SupportsSplit) -> Any: ...\n    ')

    def test_supports_strip(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f(x):\n        return x.strip()\n     ')
        self.assertTypesMatchPytd(ty, '\n      import protocols\n      from typing import Any\n      def f(x: protocols.SupportsStrip) -> Any: ...\n    ')

    def test_supports_find(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f(x):\n        return x.find("foo")\n     ')
        self.assertTypesMatchPytd(ty, '\n      import protocols\n      from typing import Any\n      def f(x: protocols.SupportsFind) -> Any: ...\n    ')

    def test_signature_template(self):
        if False:
            while True:
                i = 10
        self.assertNoCrash(self.Infer, "\n      def rearrange_proc_table(val):\n        procs = val['procs']\n        val['procs'] = dict((ix, procs[ix]) for ix in range(0, len(procs)))\n        del val['fields']\n    ")
if __name__ == '__main__':
    test_base.main()