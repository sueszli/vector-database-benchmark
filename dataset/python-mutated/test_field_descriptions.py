import strawberry

def test_field_descriptions():
    if False:
        while True:
            i = 10
    description = 'this description is super cool'
    field = strawberry.field(description=description)
    assert field.description == description