import strawberry
from strawberry.directive import DirectiveLocation
from strawberry.extensions import SchemaExtension
from strawberry.extensions.directives import DirectivesExtension, DirectivesExtensionSync

@strawberry.type
class Query:
    example: str

@strawberry.directive(locations=[DirectiveLocation.FIELD])
def uppercase(value: str) -> str:
    if False:
        while True:
            i = 10
    return value.upper()

class MyExtension(SchemaExtension):
    ...

def test_returns_empty_list_when_no_custom_directives():
    if False:
        for i in range(10):
            print('nop')
    schema = strawberry.Schema(query=Query)
    assert schema.get_extensions() == []

def test_returns_extension_passed_by_user():
    if False:
        i = 10
        return i + 15
    schema = strawberry.Schema(query=Query, extensions=[MyExtension])
    assert schema.get_extensions() == [MyExtension]

def test_returns_directives_extension_when_passing_directives():
    if False:
        i = 10
        return i + 15
    schema = strawberry.Schema(query=Query, directives=[uppercase])
    assert schema.get_extensions() == [DirectivesExtension]

def test_returns_extension_passed_by_user_and_directives_extension():
    if False:
        while True:
            i = 10
    schema = strawberry.Schema(query=Query, extensions=[MyExtension], directives=[uppercase])
    assert schema.get_extensions() == [MyExtension, DirectivesExtension]

def test_returns_directives_extension_when_passing_directives_sync():
    if False:
        return 10
    schema = strawberry.Schema(query=Query, directives=[uppercase])
    assert schema.get_extensions(sync=True) == [DirectivesExtensionSync]

def test_returns_extension_passed_by_user_and_directives_extension_sync():
    if False:
        while True:
            i = 10
    schema = strawberry.Schema(query=Query, extensions=[MyExtension], directives=[uppercase])
    assert schema.get_extensions(sync=True) == [MyExtension, DirectivesExtensionSync]