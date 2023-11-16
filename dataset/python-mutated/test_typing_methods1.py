"""Tests for the methods in the typing module."""
import textwrap
from pytype.tests import test_base
from pytype.tests import test_utils

class TypingMethodsTest(test_base.BaseTest):
    """Tests for typing.py."""

    def _check_call(self, t, expr):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import {type}\n        def f() -> {type}: ...\n      '.format(type=t))
            indented_expr = textwrap.dedent(expr).replace('\n', '\n' + ' ' * 8)
            self.Check(f'\n        import foo\n        x = foo.f()\n        {indented_expr}\n      ', pythonpath=[d.path])

    def test_text(self):
        if False:
            i = 10
            return i + 15
        self._check_call('Text', 'x.upper()')

    def test_supportsabs(self):
        if False:
            return 10
        self._check_call('SupportsAbs', 'abs(x)')

    def test_supportsround(self):
        if False:
            while True:
                i = 10
        self._check_call('SupportsRound', 'round(x)')

    def test_supportsint(self):
        if False:
            return 10
        self._check_call('SupportsInt', 'int(x); int(3)')

    def test_supportsfloat(self):
        if False:
            print('Hello World!')
        self._check_call('SupportsFloat', 'float(x); float(3.14)')

    def test_supportscomplex(self):
        if False:
            for i in range(10):
                print('nop')
        self._check_call('SupportsComplex', 'complex(x); complex(3j)')

    def test_reversible(self):
        if False:
            return 10
        self._check_call('Reversible', 'reversed(x)')

    def test_hashable(self):
        if False:
            for i in range(10):
                print('nop')
        self._check_call('Hashable', 'hash(x)')

    def test_sized(self):
        if False:
            i = 10
            return i + 15
        self._check_call('Sized', 'len(x)')

    def test_iterator(self):
        if False:
            i = 10
            return i + 15
        self._check_call('Iterator', 'next(x)')

    def test_iterable(self):
        if False:
            print('Hello World!')
        self._check_call('Iterable', 'next(iter(x))')

    def test_container(self):
        if False:
            print('Hello World!')
        self._check_call('Container', '42 in x')

    def test_io(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import IO\n        def f() -> IO[str]: ...\n      ')
            ty = self.Infer('\n        import foo\n        x = foo.f()\n        with x as fi:\n            fi.read()\n        for b in x: pass\n        a = x.fileno()\n        x.flush()\n        b = x.isatty()\n        c = x.read()\n        d = x.read(30)\n        e = x.readable()\n        f = x.readline()\n        g = x.readlines()\n        h = x.seek(0)\n        i = x.seek(0, 1)\n        j = x.seekable()\n        k = x.tell()\n        x.truncate(10)\n        m = x.writable()\n        x.write("foo")\n        x.writelines(["foo", "bar"])\n        x.close()\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import IO, List\n        fi = ...  # type: IO[str]\n        a = ...  # type: int\n        b = ...  # type: bool\n        c = ...  # type: str\n        d = ...  # type: str\n        e = ...  # type: bool\n        f = ...  # type: str\n        g = ...  # type: List[str]\n        h = ...  # type: int\n        i = ...  # type: int\n        j = ...  # type: bool\n        k = ...  # type: int\n        m = ...  # type: bool\n        x = ...  # type: IO[str]\n      ')

    def test_binary_io(self):
        if False:
            print('Hello World!')
        self._check_call('BinaryIO', 'x.read(10).upper()')

    def test_text_io(self):
        if False:
            i = 10
            return i + 15
        self._check_call('TextIO', 'x.read(10).upper()')

    def test_sequence_and_tuple(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Sequence, Tuple\n        def seq() -> Sequence[str]: ...\n        def tpl() -> Tuple[str]: ...\n      ')
            ty = self.Infer('\n        import foo\n        for seq in [foo.seq(), foo.tpl()]:\n          a = seq[0]\n          seq[0:10]\n          b = seq.index("foo")\n          c = seq.count("foo")\n          d = "foo" in seq\n          e = iter(seq)\n          f = reversed(seq)\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Iterator, List, Sequence, Tuple, Union\n        seq = ...  # type: Union[Sequence[str], Tuple[str]]\n        a = ...  # type: str\n        b = ...  # type: int\n        c = ...  # type: int\n        d = ...  # type: bool\n        e = ...  # type: Union[Iterator[str], tupleiterator[str]]\n        f = ...  # type: reversed[str]\n      ')

    def test_mutablesequence_and_list(self):
        if False:
            for i in range(10):
                print('nop')
        self.options.tweak(strict_parameter_checks=False)
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import List, MutableSequence\n        def seq() -> MutableSequence[str]: ...\n        def lst() -> List[str]: ...\n      ')
            ty = self.Infer('\n        import foo\n        for seq in [foo.seq(), foo.lst()]:\n          seq[0] = 3\n          del seq[0]\n          a = seq.append(3)\n          c = seq.insert(3, "foo")\n          d = seq.reverse()\n          e = seq.pop()\n          f = seq.pop(4)\n          g = seq.remove("foo")\n          seq[0:5] = [1,2,3]\n          b = seq.extend([1,2,3])\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Iterator, List, Sequence, Union\n        # TODO(b/159065400): Should be List[Union[int, str]]\n        seq = ...  # type: Union[list, typing.MutableSequence[Union[int, str]]]\n        a = ...  # type: None\n        b = ...  # type: None\n        c = ...  # type: None\n        d = ...  # type: None\n        e = ...  # type: Union[int, str]\n        f = ...  # type: Union[int, str]\n        g = ...  # type: None\n      ')

    def test_deque(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Deque\n        def deq() -> Deque[int]: ...\n        ')
            ty = self.Infer('\n        import foo\n        q = foo.deq()\n        q[0] = 3\n        del q[3]\n        a = q.append(3)\n        al = q.appendleft(2)\n        b = q.extend([1,2])\n        bl = q.extendleft([3,4])\n        c = q.pop()\n        cl = q.popleft()\n        d = q.rotate(3)\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Deque\n        q = ...  # type: Deque[int]\n        a = ...  # type: None\n        al = ...  # type: None\n        b = ...  # type: None\n        bl = ...  # type: None\n        c = ...  # type: int\n        cl = ...  # type: int\n        d = ...  # type: None\n      ')

    def test_mutablemapping(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import MutableMapping, TypeVar\n        K = TypeVar("K")\n        V = TypeVar("V")\n        class MyDict(MutableMapping[K, V]): ...\n        def f() -> MyDict[str, int]: ...\n      ')
            ty = self.Infer('\n        import foo\n        m = foo.f()\n        m.clear()\n        m[3j] = 3.14\n        del m["foo"]\n        a = m.pop("bar", 3j)\n        b = m.popitem()\n        c = m.setdefault("baz", 3j)\n        m.update({4j: 2.1})\n        m.update([(1, 2), (3, 4)])\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Tuple, Union\n        import foo\n        m = ...  # type: foo.MyDict[Union[complex, int, str], Union[complex, float, int]]\n        a = ...  # type: Union[complex, float, int]\n        b = ...  # type: Tuple[Union[complex, str], Union[float, int]]\n        c = ...  # type: Union[complex, float, int]\n      ')

    def test_dict_and_defaultdict(self):
        if False:
            for i in range(10):
                print('nop')
        self._check_call('DefaultDict', 'x[42j]')
        self._check_call('Dict', 'x[42j]')

    def test_abstractset(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import AbstractSet\n        def f() -> AbstractSet[str]: ...\n      ')
            ty = self.Infer('\n        import foo\n        x = foo.f()\n        a = "bar" in x\n        b = x & x\n        c = x | x\n        d = x - x\n        e = x ^ x\n        f = x.isdisjoint([1,2,3])\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import AbstractSet\n        x = ...  # type: AbstractSet[str]\n        a = ...  # type: bool\n        b = ...  # type: AbstractSet[str]\n        c = ...  # type: AbstractSet[str]\n        d = ...  # type: AbstractSet[str]\n        e = ...  # type: AbstractSet[str]\n        f = ...  # type: bool\n      ')

    def test_frozenset(self):
        if False:
            for i in range(10):
                print('nop')
        self._check_call('FrozenSet', '3 in x')

    def test_mutableset(self):
        if False:
            return 10
        self.options.tweak(strict_parameter_checks=False)
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import MutableSet\n        def f() -> MutableSet[str]: ...\n      ')
            ty = self.Infer('\n        import foo\n        x = foo.f()\n        x.add(1)\n        a = x.pop()\n        x.discard(2)\n        x.clear()\n        x.add(3j)\n        x.remove(3j)\n        b = x & {1,2,3}\n        c = x | {1,2,3}\n        d = x ^ {1,2,3}\n        e = 3 in x\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import MutableSet, Union\n        a = ...  # type: Union[int, str]\n        # TODO(b/159067449): We do a clear() after adding "int".\n        # Why does "int" still appear for b?\n        b = ...  # type: MutableSet[Union[complex, int, str]]\n        c = ...  # type: MutableSet[Union[complex, int, str]]\n        d = ...  # type: MutableSet[Union[complex, int, str]]\n        e = ...  # type: bool\n        x = ...  # type: MutableSet[Union[complex, int, str]]\n      ')

    def test_set(self):
        if False:
            print('Hello World!')
        self._check_call('Set', 'x.add(3)')

    def test_generator(self):
        if False:
            i = 10
            return i + 15
        self._check_call('Generator', '\n      next(x)\n      x.send(42)\n      x.throw(Exception())\n      x.close()\n    ')

    def test_pattern_and_match(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Pattern\n        def f() -> Pattern[str]: ...\n      ')
            ty = self.Infer('\n        import foo\n        pattern = foo.f()\n        m1 = pattern.search("foo")\n        pattern.match("foo")\n        pattern.split("foo")\n        pattern.findall("foo")[0]\n        list(pattern.finditer("foo"))[0]\n        pattern.sub("x", "x")\n        pattern.subn("x", "x")\n        assert m1\n        a = m1.pos\n        b = m1.endpos\n        c = m1.group(0)\n        d = m1.start()\n        e = m1.end()\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        import re\n        from typing import Pattern\n        a: int\n        b: int\n        c: str\n        d: int\n        e: int\n        m1: re.Match[str] | None\n        pattern: Pattern[str]\n      ')
if __name__ == '__main__':
    test_base.main()