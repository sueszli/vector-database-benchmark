"""Tests for slots."""
from pytype.tests import test_base

class SlotsTest(test_base.BaseTest):
    """Tests for __slots__."""

    def test_slots(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      class Foo:\n        __slots__ = ("foo", "bar", "baz")\n        def __init__(self):\n          self.foo = 1\n          self.bar = 2\n          self.baz = 4\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        __slots__ = ["foo", "bar", "baz"]\n        foo = ...  # type: int\n        bar = ...  # type: int\n        baz = ...  # type: int\n        def __init__(self) -> None: ...\n    ')

    def test_ambiguous_slot(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class Foo:\n        __slots__ = () if __random__ else ("foo")\n        def __init__(self):\n          self.foo = 1\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        foo = ...  # type: int\n        def __init__(self) -> None: ...\n    ')

    def test_ambiguous_slot_entry(self):
        if False:
            print('Hello World!')
        self.Check('\n      class Foo:\n        __slots__ = ("foo" if __random__ else "bar",)\n    ')

    def test_tuple_slot(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      class Foo:\n        __slots__ = ("foo", "bar")\n    ')

    def test_tuple_slot_unicode(self):
        if False:
            while True:
                i = 10
        self.Check('\n      class Foo:\n        __slots__ = (u"foo", u"bar")\n    ')

    def test_list_slot(self):
        if False:
            return 10
        ty = self.Infer('\n      class Foo:\n        __slots__ = ["foo", "bar"]\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        __slots__ = ["foo", "bar"]\n    ')

    def test_slot_with_non_strings(self):
        if False:
            for i in range(10):
                print('nop')
        (_, errors) = self.InferWithErrors('\n      class Foo:  # bad-slots[e]\n        __slots__ = (1, 2, 3)\n    ')
        self.assertErrorRegexes(errors, {'e': "Invalid __slots__ entry: '1'"})

    def test_set_slot(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      class Foo:\n        __slots__ = {"foo", "bar"}  # Note: Python actually allows this.\n      Foo().bar = 3\n    ')

    def test_slot_as_attribute(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class Foo:\n        def __init__(self):\n          self.__slots__ = ["foo"]\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        def __init__(self) -> None: ...\n    ')

    def test_slot_as_late_class_attribute(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      class Foo: pass\n      # It\'s rare to see this pattern in the wild. The only occurrence, outside\n      # of tests, seems to be https://www.gnu.org/software/gss/manual/gss.html.\n      # Note this doesn\'t actually do anything! Python ignores the next line.\n      Foo.__slots__ = ["foo"]\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        pass\n    ')

    def test_assign_attribute(self):
        if False:
            i = 10
            return i + 15
        (_, errors) = self.InferWithErrors('\n      class Foo:\n        __slots__ = ("x", "y")\n      foo = Foo()\n      foo.x = 1  # ok\n      foo.y = 2  # ok\n      foo.z = 3  # not-writable[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'z'})

    def test_object(self):
        if False:
            i = 10
            return i + 15
        (_, errors) = self.InferWithErrors('\n      object().foo = 42  # not-writable[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'object'})

    def test_any_base_class(self):
        if False:
            print('Hello World!')
        self.Check('\n      class Foo(__any_object__):\n        __slots__ = ()\n      Foo().foo = 42\n    ')

    def test_parameterized_base_class(self):
        if False:
            print('Hello World!')
        (_, errors) = self.InferWithErrors('\n      from typing import List\n      class Foo(List[int]):\n        __slots__ = ()\n      Foo().foo = 42  # not-writable[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'foo'})

    def test_empty_slots(self):
        if False:
            return 10
        (_, errors) = self.InferWithErrors('\n      class Foo:\n        __slots__ = ()\n      Foo().foo = 42  # not-writable[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'foo'})

    @test_base.skip('b/227272745')
    def test_namedtuple(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.InferWithErrors('\n      import collections\n      Foo = collections.namedtuple("_", ["a", "b", "c"])\n      foo = Foo(None, None, None)\n      foo.a = 1\n      foo.b = 2\n      foo.c = 3\n      foo.d = 4  # not-writable[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'd'})

    def test_builtin_attr(self):
        if False:
            while True:
                i = 10
        self.InferWithErrors('\n      "foo".bar = 1  # not-writable\n      u"foo".bar = 2  # not-writable\n      ().bar = 3  # not-writable\n      [].bar = 4  # not-writable\n      {}.bar = 5  # not-writable\n      set().bar = 6  # not-writable\n      frozenset().bar = 7  # not-writable\n      frozenset().bar = 8  # not-writable\n      Ellipsis.bar = 9  # not-writable\n      bytearray().bar = 10  # not-writable\n      enumerate([]).bar = 11  # not-writable\n      True.bar = 12  # not-writable\n      (42).bar = 13  # not-writable\n      (3.14).bar = 14  # not-writable\n      (3j).bar = 15  # not-writable\n      slice(1,10).bar = 16  # not-writable\n      memoryview(b"foo").bar = 17  # not-writable\n      range(10).bar = 18  # not-writable\n    ')

    def test_generator_attr(self):
        if False:
            for i in range(10):
                print('nop')
        (_, errors) = self.InferWithErrors('\n      def f(): yield 42\n      f().foo = 42  # not-writable[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'foo'})

    def test_set_attr(self):
        if False:
            print('Hello World!')
        self.Check('\n      class Foo:\n        __slots__ = ()\n        def __setattr__(self, name, value):\n          pass\n      class Bar(Foo):\n        __slots__ = ()\n      Foo().baz = 1\n      Bar().baz = 2\n    ')

    def test_descriptors(self):
        if False:
            print('Hello World!')
        self.Check('\n      class Descriptor:\n        def __set__(self, obj, cls):\n          pass\n      class Foo:\n        __slots__ = ()\n        baz = Descriptor()\n      class Bar(Foo):\n        __slots__ = ()\n      Foo().baz = 1\n      Bar().baz = 2\n    ')

    def test_name_mangling(self):
        if False:
            for i in range(10):
                print('nop')
        (_, errors) = self.InferWithErrors('\n      class Bar:\n        __slots__ = ["__baz"]\n        def __init__(self):\n          self.__baz = 42\n      class Foo(Bar):\n        __slots__ = ["__foo"]\n        def __init__(self):\n          self.__foo = 42\n          self.__baz = 42  # __baz is class-private  # not-writable[e]\n    ')
        self.assertErrorRegexes(errors, {'e': '__baz'})

    def test_union(self):
        if False:
            print('Hello World!')
        self.Check('\n      from typing import Union\n      class Foo:\n        pass\n      class Bar:\n        __slots__ = ()\n      def f(x):\n        # type: (Union[Foo, Bar]) -> None\n        if isinstance(x, Foo):\n          x.foo = 42\n    ')
if __name__ == '__main__':
    test_base.main()