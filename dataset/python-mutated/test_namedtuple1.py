"""Tests for the collections.namedtuple implementation."""
from pytype.tests import test_base
from pytype.tests import test_utils

class NamedtupleTests(test_base.BaseTest):
    """Tests for collections.namedtuple."""

    def test_basic_namedtuple(self):
        if False:
            print('Hello World!')
        self.Check('\n      import collections\n\n      X = collections.namedtuple("X", ["y", "z"])\n      a = X(y=1, z=2)\n      assert_type(a, X)\n      ', deep=False)

    def test_pytd(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      import collections\n\n      X = collections.namedtuple("X", ["y", "z"])\n      a = X(y=1, z=2)\n      ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      import collections\n      from typing import Any, NamedTuple\n\n      a: X\n\n      class X(NamedTuple):\n          y: Any\n          z: Any\n    ')

    def test_no_fields(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n        import collections\n\n        F = collections.namedtuple("F", [])\n        a = F()\n        ', deep=False)

    def test_str_args(self):
        if False:
            while True:
                i = 10
        self.Check('\n        import collections\n\n        S = collections.namedtuple("S", "a b c")\n        b = S(1, 2, 3)\n        c = (b.a, b.b, b.c)\n    ', deep=False)

    def test_str_args2(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n        import collections\n        collections.namedtuple("_", "a,b,c")(1, 2, 3)\n        ')
        self.Check('\n        import collections\n        collections.namedtuple("_", "a, b, c")(1, 2, 3)\n        ')
        self.Check('\n        import collections\n        collections.namedtuple("_", "a ,b")(1, 2)\n        ')

    def test_bad_fieldnames(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n        import collections\n        collections.namedtuple("_", ["abc", "def", "ghi"])  # invalid-namedtuple-arg\n        collections.namedtuple("_", "_")  # invalid-namedtuple-arg\n        collections.namedtuple("_", "a, 1")  # invalid-namedtuple-arg\n        collections.namedtuple("_", "a, !")  # invalid-namedtuple-arg\n        collections.namedtuple("_", "a, b, c, a")  # invalid-namedtuple-arg\n        collections.namedtuple("1", "")  # invalid-namedtuple-arg\n        ')

    def test_rename(self):
        if False:
            print('Hello World!')
        self.Check('\n        import collections\n\n        S = collections.namedtuple("S", "abc def ghi abc", rename=True)\n        a = S(1, 2, 3, 4)\n        b = a._3\n        ', deep=False)

    def test_bad_initialize(self):
        if False:
            return 10
        self.CheckWithErrors('\n        from collections import namedtuple\n\n        X = namedtuple("X", "y z")\n        a = X(1)  # missing-parameter\n        b = X(y = 2)  # missing-parameter\n        c = X(w = 3)  # wrong-keyword-args\n        d = X(y = "hello", z = 4j)  # works\n        ')

    def test_class_name(self):
        if False:
            return 10
        self.CheckWithErrors('\n        import collections\n        F = collections.namedtuple("S", [\'a\', \'b\', \'c\'])\n        a = F(1, 2, 3)\n        b = S(1, 2, 3)  # name-error\n        ')

    def test_constructors(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n        import collections\n        X = collections.namedtuple("X", "a b c")\n        g = X(1, 2, 3)\n        i = X._make((7, 8, 9))\n        j = X._make((10, 11, 12), tuple.__new__, len)\n        ')

    def test_instance_types(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n        import collections\n        X = collections.namedtuple("X", "a b c")\n        a = X._make((1, 2, 3))\n        ')

    def test_fields(self):
        if False:
            while True:
                i = 10
        self.Check('\n        import collections\n        X = collections.namedtuple("X", "a b c")\n\n        a = X(1, "2", 42.0)\n\n        a_f = a.a\n        b_f = a.b\n        c_f = a.c\n        ')

    def test_unpacking(self):
        if False:
            print('Hello World!')
        self.Check('\n        import collections\n        X = collections.namedtuple("X", "a b c")\n\n        a = X(1, "2", 42.0)\n\n        a_f, b_f, c_f = a\n        ')

    def test_bad_unpacking(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors('\n        import collections\n        X = collections.namedtuple("X", "a b c")\n\n        a = X(1, "2", 42.0)\n\n        _, _, _, too_many = a  # bad-unpacking\n        _, too_few = a  # bad-unpacking\n        ')

    def test_is_tuple_and_superclasses(self):
        if False:
            while True:
                i = 10
        'Test that a collections.namedtuple behaves like a tuple typewise.'
        self.Check('\n        import collections\n        from typing import Any, MutableSequence, Sequence, Tuple\n        X = collections.namedtuple("X", "a b c")\n\n        a = X(1, "2", 42.0)\n\n        a_tuple = a  # type: tuple\n        a_typing_tuple = a  # type: Tuple[Any, Any, Any]\n        # Collapses to just plain "tuple"\n        a_typing_tuple_ellipses = a  # type: Tuple[Any, ...]\n        a_sequence = a  # type: Sequence[Any]\n        a_iter = iter(a)  # type: tupleiterator\n        a_first = next(iter(a))\n        a_second = a[1]\n        ')

    def test_is_not_incorrect_types(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors('\n        import collections\n        from typing import Any, MutableSequence, Sequence, Tuple\n        X = collections.namedtuple("X", "a b c")\n\n        x = X(1, "2", 42.0)\n\n        x_not_a_list = x  # type: list  # annotation-type-mismatch\n        x_not_a_mutable_seq = x  # type: MutableSequence[Any]  # annotation-type-mismatch  # pylint: disable=line-too-long\n        ')

    def test_meets_protocol(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n        import collections\n        from typing import Any, Protocol\n        X = collections.namedtuple("X", ["a", "b"])\n\n        class DualVarHolder(Protocol):\n          a: Any\n          b: Any\n\n        class DualPropertyHolder(Protocol):\n          @property\n          def a(self):\n            ...\n\n          @property\n          def b(self):\n            ...\n\n        a = X(1, "2")\n        a_vars_protocol: DualVarHolder = a\n        a_property_protocol: DualPropertyHolder = a\n    ')

    def test_does_not_meet_mismatching_protocol(self):
        if False:
            return 10
        self.CheckWithErrors('\n        import collections\n        from typing import Any, Protocol\n        X = collections.namedtuple("X", ["a", "b"])\n\n        class TripleVarHolder(Protocol):\n          a: Any\n          b: Any\n          c: Any\n\n        class DualHolder_Alt(Protocol):\n          x: Any\n          y: Any\n\n        a = X(1, "2")\n        a_wrong_names: DualHolder_Alt = a  # annotation-type-mismatch\n        a_too_many: TripleVarHolder = a  # annotation-type-mismatch\n    ')

    def test_instantiate_pyi_namedtuple(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', "\n        class X(NamedTuple('X', [('y', str), ('z', int)])): ...\n      ")
            (_, errors) = self.InferWithErrors('\n        import foo\n        foo.X()  # missing-parameter[e1]\n        foo.X(0, "")  # wrong-arg-types[e2]\n        foo.X(z="", y=0)  # wrong-arg-types[e3]\n        foo.X("", 0)\n        foo.X(y="", z=0)\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e1': 'y', 'e2': 'str.*int', 'e3': 'str.*int'})

    def test_use_pyi_namedtuple(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class X(NamedTuple("X", [])): ...\n      ')
            (_, errors) = self.InferWithErrors('\n        import foo\n        foo.X()._replace()\n        foo.X().nonsense  # attribute-error[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'nonsense.*X'})

    def test_subclass_pyi_namedtuple(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class X(NamedTuple("X", [("y", int)])): ...\n      ')
            self.Check('\n        import foo\n        class Y(foo.X):\n          def __new__(cls):\n            return super(Y, cls).__new__(cls, 0)\n        Y()\n      ', pythonpath=[d.path])

    def test_varargs(self):
        if False:
            while True:
                i = 10
        self.Check('\n      import collections\n      X = collections.namedtuple("X", [])\n      args = None  # type: list\n      X(*args)\n    ')

    def test_kwargs(self):
        if False:
            while True:
                i = 10
        self.Check('\n      import collections\n      X = collections.namedtuple("X", [])\n      kwargs = None  # type: dict\n      X(**kwargs)\n    ')

    def test_name_conflict(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      import collections\n      X = collections.namedtuple("_", [])\n      Y = collections.namedtuple("_", [])\n      Z = collections.namedtuple("_", "a")\n    ', deep=False)

    def test_subclass(self):
        if False:
            while True:
                i = 10
        self.Check('\n      import collections\n      class X(collections.namedtuple("X", [])):\n        def __new__(cls, _):\n          return super(X, cls).__new__(cls)\n      a = X(1)\n      assert_type(a, X)\n    ')

    def test_subclass_replace(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      import collections\n      X = collections.namedtuple("X", "a")\n      class Y(X): pass\n      z = Y(1)._replace(a=2)\n    ')

    def test_subclass_make(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      import collections\n      X = collections.namedtuple("X", "a")\n      class Y(X): pass\n      z = Y._make([1])\n      assert_type(z, Y)\n    ')
if __name__ == '__main__':
    test_base.main()