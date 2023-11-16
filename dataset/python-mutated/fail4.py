from typing import Any
from pydantic import BaseModel, root_validator, validate_call

@validate_call
def foo(a: int, *, c: str='x') -> str:
    if False:
        return 10
    return c * a
x: str = foo(1, c='hello')
foo('x')
foo(1, c=1)
foo(1, 2)
foo(1, d=2)
callable(foo.raw_function)

@validate_call
def bar() -> str:
    if False:
        while True:
            i = 10
    return 'x'
y: int = bar()

class Model(BaseModel):

    @root_validator()
    @classmethod
    def validate_1(cls, values: Any) -> Any:
        if False:
            while True:
                i = 10
        return values

    @root_validator(pre=True, skip_on_failure=True)
    @classmethod
    def validate_2(cls, values: Any) -> Any:
        if False:
            while True:
                i = 10
        return values

    @root_validator(pre=False)
    @classmethod
    def validate_3(cls, values: Any) -> Any:
        if False:
            return 10
        return values
Model.non_existent_attribute