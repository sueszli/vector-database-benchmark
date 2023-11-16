import strawberry

def test_type_add_type_definition_with_fields():
    if False:
        print('Hello World!')

    @strawberry.type
    class Query:
        name: str
        age: int
    definition = Query.__strawberry_definition__
    assert definition.name == 'Query'
    assert len(definition.fields) == 2
    assert definition.fields[0].python_name == 'name'
    assert definition.fields[0].type is str
    assert definition.fields[1].python_name == 'age'
    assert definition.fields[1].type is int

def test_passing_nothing_to_fields():
    if False:
        i = 10
        return i + 15

    @strawberry.type
    class Query:
        name: str = strawberry.field()
        age: int = strawberry.field()
    definition = Query.__strawberry_definition__
    assert definition.name == 'Query'
    assert len(definition.fields) == 2
    assert definition.fields[0].python_name == 'name'
    assert definition.fields[0].type is str
    assert definition.fields[1].python_name == 'age'
    assert definition.fields[1].type is int