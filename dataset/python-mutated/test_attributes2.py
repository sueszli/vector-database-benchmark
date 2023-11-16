"""Test instance and class attributes."""
from pytype.tests import test_base

class TestStrictNone(test_base.BaseTest):
    """Tests for strict attribute checking on None."""

    def test_explicit_none(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      from typing import Optional\n      def f(x: Optional[str]):\n        return x.upper()  # attribute-error[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'upper.*None'})

    def test_closure(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      from typing import Optional\n      d = ...  # type: Optional[dict]\n      if d:\n        formatter = lambda x: d.get(x, '')\n      else:\n        formatter = lambda x: ''\n      formatter('key')\n    ")

    def test_overwrite_global(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors("\n      from typing import Optional\n      d = ...  # type: Optional[dict]\n      if d:\n        formatter = lambda x: d.get(x, '')  # attribute-error[e]\n      else:\n        formatter = lambda x: ''\n      d = None\n      formatter('key')  # line 8\n    ")
        self.assertErrorRegexes(errors, {'e': 'get.*None'})

class TestAttributes(test_base.BaseTest):
    """Tests for attributes."""

    def test_attr_on_optional(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      from typing import Optional\n      def f(x: Optional[str]):\n        return x.upper()  # attribute-error[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'upper.*None'})

    def test_any_annotation(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import Any\n      def f(x: Any):\n        if __random__:\n          x = 42\n        x.upper()\n    ')

    def test_non_init_annotation(self):
        if False:
            print('Hello World!')
        (ty, errors) = self.InferWithErrors("\n      from typing import List\n      class Foo:\n        def __init__(self):\n          # This annotation should be used when inferring the attribute type.\n          self.x = []  # type: List[int]\n        def f1(self):\n          # This annotation should be applied to the attribute value but ignored\n          # when inferring the attribute type.\n          self.x = []  # type: List[str]\n          return self.x\n        def f2(self):\n          # This assignment should be checked against the __init__ annotation.\n          self.x = ['']  # annotation-type-mismatch[e]\n        def f3(self):\n          # The return type should reflect all assignments, even ones that\n          # violate the __init__ annotation.\n          return self.x\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import List, Union\n      class Foo:\n        x: List[int]\n        def __init__(self) -> None: ...\n        def f1(self) -> List[str]: ...\n        def f2(self) -> None: ...\n        def f3(self) -> List[Union[int, str]]: ...\n    ')
        self.assertErrorRegexes(errors, {'e': 'Annotation: List\\[int\\].*Assignment: List\\[str\\]'})

    def test_set_attribute_in_other_class(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from typing import Set\n      class Foo:\n        def __init__(self):\n          self.x: Set = set()\n          self.y = set()\n      class Bar:\n        def __init__(self):\n          self.foo = self.get_foo()\n        def get_foo() -> Foo:\n          if __random__:\n            return Foo()\n          else:\n            return Foo()\n        def fix_foo(self):\n          self.foo.extra = set()\n    ')

    def test_base_class_union(self):
        if False:
            print('Hello World!')
        with self.DepTree([('foo.pyi', '\n      class A:\n        x: str\n      class B:\n        x: str\n      class C1:\n        C: type[A]\n      class C2:\n        C: type[B]\n      def f() -> C1 | C2: ...\n    '), ('bar.py', '\n       import foo\n       class Bar(foo.f().C):\n         pass\n       assert_type(Bar().x, str)\n    ')]):
            self.Check('\n        import bar\n        assert_type(bar.Bar().x, str)\n      ')

class TestAttributesPython3FeatureTest(test_base.BaseTest):
    """Tests for attributes over target code using Python 3 features."""

    def test_empty_type_parameter_instance(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      args = {}\n      for x, y in sorted(args.items()):\n        x.values\n    ')

    def test_type_parameter_instance_multiple_bindings(self):
        if False:
            print('Hello World!')
        (_, errors) = self.InferWithErrors('\n      class A:\n        values = 42\n      args = {A() if __random__ else True: ""}\n      for x, y in sorted(args.items()):\n        x.values  # attribute-error[e]\n    ')
        self.assertErrorRegexes(errors, {'e': "'values' on bool"})

    def test_type_parameter_instance_set_attr(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      class Foo:\n        pass\n      class Bar:\n        def bar(self):\n          d = {42: Foo()}\n          for _, foo in sorted(d.items()):\n            foo.x = 42\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        x = ...  # type: int\n      class Bar:\n        def bar(self) -> None: ...\n    ')

    def test_type_parameter_instance(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      class A:\n        values = 42\n      args = {A(): ""}\n      for x, y in sorted(args.items()):\n        z = x.values\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict\n      class A:\n        values = ...  # type: int\n      args = ...  # type: Dict[A, str]\n      x = ...  # type: A\n      y = ...  # type: str\n      z = ...  # type: int\n    ')

    def test_filter_subclass_attribute(self):
        if False:
            print('Hello World!')
        self.Check('\n      from typing import List\n\n      class NamedObject:\n        name = ...  # type: str\n      class UnnamedObject:\n        pass\n      class ObjectHolder:\n        named = ...  # type: NamedObject\n        unnamed = ...  # type: UnnamedObject\n\n      class Base:\n        def __init__(self):\n          self.objects = []  # type: List\n\n      class Foo(Base):\n        def __init__(self, holder: ObjectHolder):\n          Base.__init__(self)\n          self.objects.append(holder.named)\n        def get_name(self):\n          return self.objects[0].name\n\n      class Bar(Base):\n        def __init__(self, holder: ObjectHolder):\n          Base.__init__(self)\n          self.objects = []\n          self.objects.append(holder.unnamed)\n    ')

    @test_base.skip('Needs vm._get_iter() to iterate over individual bindings.')
    def test_metaclass_iter(self):
        if False:
            while True:
                i = 10
        self.Check('\n      class Meta(type):\n        def __iter__(cls):\n          return iter([])\n      class Foo(metaclass=Meta):\n        def __iter__(self):\n          return iter([])\n      for _ in Foo:\n        pass\n    ')

    @test_base.skip('Needs better handling of __getitem__ in vm._get_iter().')
    def test_metaclass_getitem(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      class Meta(type):\n        def __getitem__(cls, x):\n          return 0\n      class Foo(metaclass=Meta):\n        def __getitem__(self, x):\n          return 0\n      for _ in Foo:\n        pass\n    ')

    def test_check_variable_annotation(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors("\n      class Foo:\n        x: int\n        def foo(self):\n          self.x = 'hello, world'  # annotation-type-mismatch[e]\n    ")
        self.assertErrorRegexes(errors, {'e': 'Annotation: int.*Assignment: str'})

    def test_pep526_annotation(self):
        if False:
            return 10
        (ty, _) = self.InferWithErrors('\n      class Foo:\n        def __init__(self):\n          self.x: int = None\n        def do_something(self, x: str):\n          self.x = x  # annotation-type-mismatch\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        x: int\n        def __init__(self) -> None: ...\n        def do_something(self, x: str) -> None: ...\n    ')

    def test_inherit_declared_attribute(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class Foo:\n        x: int\n      class Bar(Foo):\n        def f(self):\n          return self.x\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        x: int\n      class Bar(Foo):\n        def f(self) -> int: ...\n    ')

    def test_redeclare_inherited_attribute(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('foo.pyi', '\n        class Foo:\n          x: int\n    ')]):
            self.Check('\n        import foo\n        class Bar(foo.Foo):\n          x: str\n        assert_type(Bar.x, str)\n      ')

    def test_attribute_access(self):
        if False:
            print('Hello World!')
        self.Check('\n      from typing import Dict, Type, Union\n      class C0:\n        def __init__(self):\n          self.x = 42\n        def f(self):\n          return self.x\n      class C1: pass\n      class C2: pass\n      mapping: Dict[Type[Union[C1, C2]], C0]\n      assert_type(mapping[C1].f(), int)\n    ')
if __name__ == '__main__':
    test_base.main()