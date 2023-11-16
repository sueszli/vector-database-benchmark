from __future__ import annotations
import dataclasses
import sys
from typing import Dict, List, Type
from strawberry.annotation import StrawberryAnnotation
from strawberry.exceptions import FieldWithResolverAndDefaultFactoryError, FieldWithResolverAndDefaultValueError, PrivateStrawberryFieldError
from strawberry.field import StrawberryField
from strawberry.private import is_private
from strawberry.type import has_object_definition
from strawberry.unset import UNSET

def _get_fields(cls: Type) -> List[StrawberryField]:
    if False:
        i = 10
        return i + 15
    "Get all the strawberry fields off a strawberry.type cls\n\n    This function returns a list of StrawberryFields (one for each field item), while\n    also paying attention the name and typing of the field.\n\n    StrawberryFields can be defined on a strawberry.type class as either a dataclass-\n    style field or using strawberry.field as a decorator.\n\n    >>> import strawberry\n    >>> @strawberry.type\n    ... class Query:\n    ...     type_1a: int = 5\n    ...     type_1b: int = strawberry.field(...)\n    ...     type_1c: int = strawberry.field(resolver=...)\n    ...\n    ...     @strawberry.field\n    ...     def type_2(self) -> int:\n    ...         ...\n\n    Type #1:\n        A pure dataclass-style field. Will not have a StrawberryField; one will need to\n        be created in this function. Type annotation is required.\n\n    Type #2:\n        A field defined using @strawberry.field as a decorator around the resolver. The\n        resolver must be type-annotated.\n\n    The StrawberryField.python_name value will be assigned to the field's name on the\n    class if one is not set by either using an explicit strawberry.field(name=...) or by\n    passing a named function (i.e. not an anonymous lambda) to strawberry.field\n    (typically as a decorator).\n    "
    fields: Dict[str, StrawberryField] = {}
    for base in cls.__bases__:
        if has_object_definition(base):
            base_fields = {field.python_name: field for field in base.__strawberry_definition__.fields}
            fields = {**fields, **base_fields}
    origins: Dict[str, type] = {field_name: cls for field_name in cls.__annotations__}
    for base in cls.__mro__:
        if has_object_definition(base):
            for field in base.__strawberry_definition__.fields:
                if field.python_name in base.__annotations__:
                    origins.setdefault(field.name, base)
    for field in dataclasses.fields(cls):
        if isinstance(field, StrawberryField):
            if is_private(field.type):
                raise PrivateStrawberryFieldError(field.python_name, cls)
            if field.default is not dataclasses.MISSING and field.default is not UNSET and (field.base_resolver is not None):
                raise FieldWithResolverAndDefaultValueError(field.python_name, cls.__name__)
            default_factory = getattr(field, 'default_factory', None)
            if default_factory is not dataclasses.MISSING and default_factory is not UNSET and (field.base_resolver is not None):
                raise FieldWithResolverAndDefaultFactoryError(field.python_name, cls.__name__)
            field.origin = field.origin or cls
            if isinstance(field.type_annotation, StrawberryAnnotation) and field.type_annotation.namespace is None:
                field.type_annotation.set_namespace_from_field(field)
        else:
            if is_private(field.type):
                continue
            origin = origins.get(field.name, cls)
            module = sys.modules[origin.__module__]
            field = StrawberryField(python_name=field.name, graphql_name=None, type_annotation=StrawberryAnnotation(annotation=field.type, namespace=module.__dict__), origin=origin, default=getattr(cls, field.name, dataclasses.MISSING))
        field_name = field.python_name
        assert_message = 'Field must have a name by the time the schema is generated'
        assert field_name is not None, assert_message
        fields[field_name] = field
    return list(fields.values())