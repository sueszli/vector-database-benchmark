import pytest
import strawberry
from strawberry.exceptions.unresolved_field_type import UnresolvedFieldTypeError

@pytest.mark.raises_strawberry_exception(UnresolvedFieldTypeError, match="Could not resolve the type of 'user'. Check that the class is accessible from the global module scope.")
def test_unresolved_field_fails():
    if False:
        while True:
            i = 10

    @strawberry.type
    class Query:
        user: 'User'
    strawberry.Schema(query=Query)

@pytest.mark.raises_strawberry_exception(UnresolvedFieldTypeError, match="Could not resolve the type of 'user'. Check that the class is accessible from the global module scope.")
def test_unresolved_field_with_resolver_fails():
    if False:
        for i in range(10):
            print('nop')

    @strawberry.type
    class Query:

        @strawberry.field
        def user(self) -> 'User':
            if False:
                return 10
            ...
    strawberry.Schema(query=Query)