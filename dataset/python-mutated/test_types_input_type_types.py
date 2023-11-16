import unittest
from typing import Optional
from pulumi._types import input_type_types
import pulumi

@pulumi.input_type
class Foo:

    @property
    @pulumi.getter()
    def bar(self) -> pulumi.Input[str]:
        if False:
            for i in range(10):
                print('nop')
        ...

@pulumi.input_type
class MySimpleInputType:
    a: str
    b: Optional[str]
    c: pulumi.Input[str]
    d: Optional[pulumi.Input[str]]
    e: Foo
    f: Optional[Foo]
    g: pulumi.Input[Foo]
    h: Optional[pulumi.Input[Foo]]
    i: pulumi.InputType[Foo]
    j: Optional[pulumi.InputType[Foo]]
    k: pulumi.Input[pulumi.InputType[Foo]]
    l: Optional[pulumi.Input[pulumi.InputType[Foo]]]

@pulumi.input_type
class MyPropertiesInputType:

    @property
    @pulumi.getter()
    def a(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        ...

    @property
    @pulumi.getter()
    def b(self) -> Optional[str]:
        if False:
            return 10
        ...

    @property
    @pulumi.getter()
    def c(self) -> pulumi.Input[str]:
        if False:
            print('Hello World!')
        ...

    @property
    @pulumi.getter()
    def d(self) -> Optional[pulumi.Input[str]]:
        if False:
            while True:
                i = 10
        ...

    @property
    @pulumi.getter()
    def e(self) -> Foo:
        if False:
            i = 10
            return i + 15
        ...

    @property
    @pulumi.getter()
    def f(self) -> Optional[Foo]:
        if False:
            i = 10
            return i + 15
        ...

    @property
    @pulumi.getter()
    def g(self) -> pulumi.Input[Foo]:
        if False:
            return 10
        ...

    @property
    @pulumi.getter()
    def h(self) -> Optional[pulumi.Input[Foo]]:
        if False:
            for i in range(10):
                print('nop')
        ...

    @property
    @pulumi.getter()
    def i(self) -> pulumi.InputType[Foo]:
        if False:
            print('Hello World!')
        ...

    @property
    @pulumi.getter()
    def j(self) -> Optional[pulumi.InputType[Foo]]:
        if False:
            i = 10
            return i + 15
        ...

    @property
    @pulumi.getter()
    def k(self) -> pulumi.Input[pulumi.InputType[Foo]]:
        if False:
            i = 10
            return i + 15
        ...

    @property
    @pulumi.getter()
    def l(self) -> Optional[pulumi.Input[pulumi.InputType[Foo]]]:
        if False:
            print('Hello World!')
        ...

class InputTypeTypesTests(unittest.TestCase):

    def test_input_type_types(self):
        if False:
            for i in range(10):
                print('nop')
        expected = {'a': str, 'b': str, 'c': str, 'd': str, 'e': Foo, 'f': Foo, 'g': Foo, 'h': Foo, 'i': Foo, 'j': Foo, 'k': Foo, 'l': Foo}
        self.assertEqual(expected, input_type_types(MySimpleInputType))
        self.assertEqual(expected, input_type_types(MyPropertiesInputType))