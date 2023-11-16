from typing import Any, Optional, Type, TypeVar
_T = TypeVar('_T')
_TClass = TypeVar('_TClass')

def none_throws(optional: Optional[_T], message: str='Unexpected `None`') -> _T:
    if False:
        return 10
    'Convert an optional to its value. Raises an `AssertionError` if the\n    value is `None`'
    if optional is None:
        raise AssertionError(message)
    return optional

def assert_is_instance(obj: object, cls: Type[_TClass]) -> _TClass:
    if False:
        while True:
            i = 10
    'Assert that the given object is an instance of the given class. Raises a\n    `TypeError` if not.'
    if not isinstance(obj, cls):
        raise TypeError(f'obj is not an instance of cls: obj={obj} cls={cls}')
    return obj

def safe_cast(new_type: Type[_T], value: Any) -> _T:
    if False:
        i = 10
        return i + 15
    "safe_cast will change the type checker's inference of x if it was\n    already a subtype of what we are casting to, and error otherwise."
    return value