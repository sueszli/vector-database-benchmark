import types
from typing import List, Optional, Sequence, Type
import strawberry
from strawberry.field import StrawberryField

def create_type(name: str, fields: List[StrawberryField], is_input: bool=False, is_interface: bool=False, description: Optional[str]=None, directives: Optional[Sequence[object]]=(), extend: bool=False) -> Type:
    if False:
        i = 10
        return i + 15
    'Create a Strawberry type from a list of StrawberryFields\n\n    >>> @strawberry.field\n    >>> def hello(info) -> str:\n    >>>     return "World"\n    >>>\n    >>> Query = create_type(name="Query", fields=[hello])\n    '
    if not fields:
        raise ValueError(f'''Can't create type "{name}" with no fields''')
    namespace = {}
    annotations = {}
    for field in fields:
        if not isinstance(field, StrawberryField):
            raise TypeError('Field is not an instance of StrawberryField')
        if field.python_name is None:
            raise ValueError("Field doesn't have a name. Fields passed to `create_type` must define a name by passing the `name` argument to `strawberry.field`.")
        namespace[field.python_name] = field
        annotations[field.python_name] = field.type
    namespace['__annotations__'] = annotations
    cls = types.new_class(name, (), {}, lambda ns: ns.update(namespace))
    return strawberry.type(cls, is_input=is_input, is_interface=is_interface, description=description, directives=directives, extend=extend)