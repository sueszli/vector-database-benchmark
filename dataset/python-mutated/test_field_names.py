import strawberry

def test_field_name_standard():
    if False:
        for i in range(10):
            print('nop')
    standard_field = strawberry.field()
    assert standard_field.python_name is None
    assert standard_field.graphql_name is None

def test_field_name_standard_on_schema():
    if False:
        print('Hello World!')

    @strawberry.type()
    class Query:
        normal_field: int
    [field] = Query.__strawberry_definition__.fields
    assert field.python_name == 'normal_field'
    assert field.graphql_name is None

def test_field_name_override():
    if False:
        for i in range(10):
            print('nop')
    field_name = 'override'
    standard_field = strawberry.field(name=field_name)
    assert standard_field.python_name is None
    assert standard_field.graphql_name == field_name

def test_field_name_override_with_schema():
    if False:
        while True:
            i = 10
    field_name = 'override_name'

    @strawberry.type()
    class Query:
        override_field: bool = strawberry.field(name=field_name)
    [field] = Query.__strawberry_definition__.fields
    assert field.python_name == 'override_field'
    assert field.graphql_name == field_name