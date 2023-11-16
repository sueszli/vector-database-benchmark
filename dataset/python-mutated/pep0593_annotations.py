from typing import Annotated

class Foo:
    pass

class A:
    pass

def annotated_function_params(basic: Annotated[str, Foo()], obj: A, annotated_obj: Annotated[A, Foo()]):
    if False:
        i = 10
        return i + 15
    basic
    obj
    annotated_obj