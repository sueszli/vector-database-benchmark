from typing import Optional
from returns.maybe import Maybe, Nothing, Some

def test_bind_some():
    if False:
        i = 10
        return i + 15
    'Ensures that bind works correctly.'

    def factory(inner_value: int) -> Maybe[int]:
        if False:
            print('Hello World!')
        return Some(inner_value * 2)
    input_value = 5
    bound = Some(input_value).bind(factory)
    assert bound == factory(input_value)
    assert str(bound) == '<Some: 10>'

def test_bind_optional():
    if False:
        i = 10
        return i + 15
    'Ensures that bind_optional works correctly.'

    def factory(inner_value: int) -> Optional[int]:
        if False:
            while True:
                i = 10
        return inner_value if inner_value else None
    assert Some(1).bind_optional(factory) == Some(1)
    assert Some(0).bind_optional(factory) == Nothing
    assert Nothing.bind_optional(factory) == Nothing