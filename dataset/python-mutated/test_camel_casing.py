import strawberry
from strawberry.schema.config import StrawberryConfig

def test_camel_case_is_on_by_default():
    if False:
        i = 10
        return i + 15

    @strawberry.type
    class Query:
        example_field: str = 'Example'
    schema = strawberry.Schema(query=Query)
    query = '\n        {\n            __type(name: "Query") {\n                fields {\n                    name\n                }\n            }\n        }\n    '
    result = schema.execute_sync(query, root_value=Query())
    assert not result.errors
    assert result.data['__type']['fields'] == [{'name': 'exampleField'}]

def test_can_set_camel_casing():
    if False:
        print('Hello World!')

    @strawberry.type
    class Query:
        example_field: str = 'Example'
    schema = strawberry.Schema(query=Query, config=StrawberryConfig(auto_camel_case=True))
    query = '\n        {\n            __type(name: "Query") {\n                fields {\n                    name\n                }\n            }\n        }\n    '
    result = schema.execute_sync(query, root_value=Query())
    assert not result.errors
    assert result.data['__type']['fields'] == [{'name': 'exampleField'}]

def test_can_set_camel_casing_to_false():
    if False:
        i = 10
        return i + 15

    @strawberry.type
    class Query:
        example_field: str = 'Example'
    schema = strawberry.Schema(query=Query, config=StrawberryConfig(auto_camel_case=False))
    query = '\n        {\n            __type(name: "Query") {\n                fields {\n                    name\n                }\n            }\n        }\n    '
    result = schema.execute_sync(query, root_value=Query())
    assert not result.errors
    assert result.data['__type']['fields'] == [{'name': 'example_field'}]

def test_can_set_camel_casing_to_false_uses_name():
    if False:
        return 10

    @strawberry.type
    class Query:
        example_field: str = strawberry.field(name='exampleField')
    schema = strawberry.Schema(query=Query, config=StrawberryConfig(auto_camel_case=False))
    query = '\n        {\n            __type(name: "Query") {\n                fields {\n                    name\n                }\n            }\n        }\n    '
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data['__type']['fields'] == [{'name': 'exampleField'}]

def test_can_set_camel_casing_to_false_uses_name_field_decorator():
    if False:
        print('Hello World!')

    @strawberry.type
    class Query:

        @strawberry.field(name='exampleField')
        def example_field(self) -> str:
            if False:
                i = 10
                return i + 15
            return 'ABC'
    schema = strawberry.Schema(query=Query, config=StrawberryConfig(auto_camel_case=False))
    query = '\n        {\n            __type(name: "Query") {\n                fields {\n                    name\n                }\n            }\n        }\n    '
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data['__type']['fields'] == [{'name': 'exampleField'}]

def test_camel_case_is_on_by_default_arguments():
    if False:
        i = 10
        return i + 15

    @strawberry.type
    class Query:

        @strawberry.field
        def example_field(self, example_input: str) -> str:
            if False:
                while True:
                    i = 10
            return example_input
    schema = strawberry.Schema(query=Query)
    query = '\n        {\n            __type(name: "Query") {\n                fields {\n                    name\n                    args { name }\n                }\n            }\n        }\n    '
    result = schema.execute_sync(query, root_value=Query())
    assert not result.errors
    assert result.data['__type']['fields'] == [{'args': [{'name': 'exampleInput'}], 'name': 'exampleField'}]

def test_can_turn_camel_case_off_arguments():
    if False:
        i = 10
        return i + 15

    @strawberry.type
    class Query:

        @strawberry.field
        def example_field(self, example_input: str) -> str:
            if False:
                return 10
            return example_input
    schema = strawberry.Schema(query=Query, config=StrawberryConfig(auto_camel_case=False))
    query = '\n        {\n            __type(name: "Query") {\n                fields {\n                    name\n                    args { name }\n                }\n            }\n        }\n    '
    result = schema.execute_sync(query, root_value=Query())
    assert not result.errors
    assert result.data['__type']['fields'] == [{'args': [{'name': 'example_input'}], 'name': 'example_field'}]

def test_can_turn_camel_case_off_arguments_conversion_works():
    if False:
        i = 10
        return i + 15

    @strawberry.type
    class Query:

        @strawberry.field
        def example_field(self, example_input: str) -> str:
            if False:
                while True:
                    i = 10
            return example_input
    schema = strawberry.Schema(query=Query, config=StrawberryConfig(auto_camel_case=False))
    query = '\n        {\n            example_field(example_input: "Hello world")\n        }\n    '
    result = schema.execute_sync(query, root_value=Query())
    assert not result.errors
    assert result.data['example_field'] == 'Hello world'