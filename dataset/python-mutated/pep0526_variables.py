"""
PEP 526 introduced a way of using type annotations on variables.
"""
import typing
asdf = ''
asdf: int
asdf
direct: int = NOT_DEFINED
direct
with_typing_module: typing.List[float] = NOT_DEFINED
with_typing_module[0]
somelist = [1, 2, 3, 'A', 'A']
element: int
for element in somelist:
    element
test_string: str = NOT_DEFINED
test_string
char: str
for char in NOT_DEFINED:
    char

class Foo:
    bar: int
    baz: typing.ClassVar[str]
Foo.bar
Foo().bar
Foo.baz
Foo().baz

class VarClass:
    var_instance1: int = ''
    var_instance2: float
    var_class1: typing.ClassVar[str] = 1
    var_class2: typing.ClassVar[bytes]
    var_class3 = None

    def __init__(self):
        if False:
            print('Hello World!')
        d.var_instance1
        d.var_instance2
        d.var_class1
        d.var_class2
        d.int
        self.var_

class VarClass2(VarClass):
    var_class3: typing.ClassVar[int]

    def __init__(self):
        if False:
            print('Hello World!')
        self.var_class3
VarClass.var_
VarClass.var_instance1
VarClass.var_instance2
VarClass.var_class1
VarClass.var_class2
VarClass.int
d = VarClass()
d.var_
d.var_instance1
d.var_instance2
d.var_class1
d.var_class2
d.int
import dataclasses

@dataclasses.dataclass
class DC:
    name: int = 1
DC().name