import textwrap
from typing import Any
import pytest
import strawberry
from strawberry.annotation import StrawberryAnnotation
from strawberry.exceptions import FieldWithResolverAndDefaultFactoryError, FieldWithResolverAndDefaultValueError
from strawberry.extensions.field_extension import FieldExtension
from strawberry.field import StrawberryField

def test_field_with_resolver_default():
    if False:
        while True:
            i = 10
    with pytest.raises(FieldWithResolverAndDefaultValueError):

        @strawberry.type
        class Query:

            @strawberry.field(default='potato')
            def fruit(self) -> str:
                if False:
                    print('Hello World!')
                return 'tomato'

def test_field_with_separate_resolver_default():
    if False:
        return 10
    with pytest.raises(FieldWithResolverAndDefaultValueError):

        def gun_resolver() -> str:
            if False:
                return 10
            return 'revolver'

        @strawberry.type
        class Query:
            weapon: str = strawberry.field(default='sword', resolver=gun_resolver)

def test_field_with_resolver_default_factory():
    if False:
        while True:
            i = 10
    with pytest.raises(FieldWithResolverAndDefaultFactoryError):

        @strawberry.type
        class Query:

            @strawberry.field(default_factory=lambda : 'steel')
            def metal(self) -> str:
                if False:
                    print('Hello World!')
                return 'iron'

def test_extension_changing_field_return_value():
    if False:
        i = 10
        return i + 15
    "Ensure that field extensions can change the field's return type."

    class ChangeReturnTypeExtension(FieldExtension):

        def apply(self, field: StrawberryField) -> None:
            if False:
                i = 10
                return i + 15
            field.type_annotation = StrawberryAnnotation.from_annotation(int)

        def resolve(self, next_, source, info, **kwargs: Any):
            if False:
                i = 10
                return i + 15
            return next_(source, info, **kwargs)

    @strawberry.type
    class Query:

        @strawberry.field(extensions=[ChangeReturnTypeExtension()])
        def test_changing_return_type(self) -> bool:
            if False:
                print('Hello World!')
            ...
    schema = strawberry.Schema(query=Query)
    expected = '      type Query {\n        testChangingReturnType: Int!\n      }\n    '
    assert str(schema) == textwrap.dedent(expected).strip()