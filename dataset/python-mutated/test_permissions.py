from typing import Any
import strawberry
from strawberry.permission import BasePermission
from strawberry.types import Info

def test_permission_classes_basic_fields():
    if False:
        return 10

    class IsAuthenticated(BasePermission):
        message = 'User is not authenticated'

        def has_permission(self, source: Any, info: Info, **kwargs: Any) -> bool:
            if False:
                i = 10
                return i + 15
            return False

    @strawberry.type
    class Query:
        user: str = strawberry.field(permission_classes=[IsAuthenticated])
    definition = Query.__strawberry_definition__
    assert definition.name == 'Query'
    assert len(definition.fields) == 1
    assert definition.fields[0].python_name == 'user'
    assert definition.fields[0].graphql_name is None
    assert definition.fields[0].permission_classes == [IsAuthenticated]

def test_permission_classes():
    if False:
        print('Hello World!')

    class IsAuthenticated(BasePermission):
        message = 'User is not authenticated'

        def has_permission(self, source: Any, info: Info, **kwargs: Any) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            return False

    @strawberry.type
    class Query:

        @strawberry.field(permission_classes=[IsAuthenticated])
        def user(self) -> str:
            if False:
                print('Hello World!')
            return 'patrick'
    definition = Query.__strawberry_definition__
    assert definition.name == 'Query'
    assert len(definition.fields) == 1
    assert definition.fields[0].python_name == 'user'
    assert definition.fields[0].graphql_name is None
    assert definition.fields[0].permission_classes == [IsAuthenticated]