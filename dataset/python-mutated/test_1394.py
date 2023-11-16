from ...types import ObjectType, Schema, String, NonNull

class Query(ObjectType):
    hello = String(input=NonNull(String))

    def resolve_hello(self, info, input):
        if False:
            for i in range(10):
                print('nop')
        if input == 'nothing':
            return None
        return f'Hello {input}!'
schema = Schema(query=Query)

def test_required_input_provided():
    if False:
        while True:
            i = 10
    '\n    Test that a required argument works when provided.\n    '
    input_value = 'Potato'
    result = schema.execute('{ hello(input: "%s") }' % input_value)
    assert not result.errors
    assert result.data == {'hello': 'Hello Potato!'}

def test_required_input_missing():
    if False:
        print('Hello World!')
    '\n    Test that a required argument raised an error if not provided.\n    '
    result = schema.execute('{ hello }')
    assert result.errors
    assert len(result.errors) == 1
    assert result.errors[0].message == "Field 'hello' argument 'input' of type 'String!' is required, but it was not provided."