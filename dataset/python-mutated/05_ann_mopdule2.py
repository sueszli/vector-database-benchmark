"""
Some correct syntax for variable annotation here.
More examples are in test_grammar and test_parser.
"""
from typing import no_type_check, ClassVar
i: int = 1
j: int
x: float = i / 10

def f():
    if False:
        for i in range(10):
            print('nop')

    class C:
        ...
    return C()
f().new_attr: object = object()

class C:

    def __init__(self, x: int) -> None:
        if False:
            while True:
                i = 10
        self.x = x
c = C(5)
c.new_attr: int = 10
__annotations__ = {}

@no_type_check
class NTC:

    def meth(self, param: complex) -> None:
        if False:
            for i in range(10):
                print('nop')
        ...

class CV:
    var: ClassVar['CV']
CV.var = CV()