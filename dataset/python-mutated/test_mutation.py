import dataclasses
import typing
from textwrap import dedent
import strawberry
from strawberry.unset import UNSET

def test_mutation():
    if False:
        while True:
            i = 10

    @strawberry.type
    class Query:
        hello: str = 'Hello'

    @strawberry.type
    class Mutation:

        @strawberry.mutation
        def say(self) -> str:
            if False:
                while True:
                    i = 10
            return 'Hello!'
    schema = strawberry.Schema(query=Query, mutation=Mutation)
    query = 'mutation { say }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data['say'] == 'Hello!'

def test_mutation_with_input_type():
    if False:
        return 10

    @strawberry.input
    class SayInput:
        name: str
        age: int

    @strawberry.type
    class Query:
        hello: str = 'Hello'

    @strawberry.type
    class Mutation:

        @strawberry.mutation
        def say(self, input: SayInput) -> str:
            if False:
                for i in range(10):
                    print('nop')
            return f'Hello {input.name} of {input.age} years old!'
    schema = strawberry.Schema(query=Query, mutation=Mutation)
    query = 'mutation { say(input: { name: "Patrick", age: 10 }) }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data['say'] == 'Hello Patrick of 10 years old!'

def test_mutation_reusing_input_types():
    if False:
        i = 10
        return i + 15

    @strawberry.input
    class SayInput:
        name: str
        age: int

    @strawberry.type
    class Query:
        hello: str = 'Hello'

    @strawberry.type
    class Mutation:

        @strawberry.mutation
        def say(self, input: SayInput) -> str:
            if False:
                return 10
            return f'Hello {input.name} of {input.age} years old!'

        @strawberry.mutation
        def say2(self, input: SayInput) -> str:
            if False:
                i = 10
                return i + 15
            return f'Hello {input.name} of {input.age}!'
    schema = strawberry.Schema(query=Query, mutation=Mutation)
    query = 'mutation { say2(input: { name: "Patrick", age: 10 }) }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data['say2'] == 'Hello Patrick of 10!'

def test_unset_types():
    if False:
        for i in range(10):
            print('nop')

    @strawberry.type
    class Query:
        hello: str = 'Hello'

    @strawberry.input
    class InputExample:
        name: str
        age: typing.Optional[int] = UNSET

    @strawberry.type
    class Mutation:

        @strawberry.mutation
        def say(self, name: typing.Optional[str]=UNSET) -> str:
            if False:
                return 10
            if name is UNSET:
                return 'Name is unset'
            return f'Hello {name}!'

        @strawberry.mutation
        def say_age(self, input: InputExample) -> str:
            if False:
                while True:
                    i = 10
            age = 'unset' if input.age is UNSET else input.age
            return f'Hello {input.name} of age {age}!'
    schema = strawberry.Schema(query=Query, mutation=Mutation)
    query = 'mutation { say sayAge(input: { name: "P"}) }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data['say'] == 'Name is unset'
    assert result.data['sayAge'] == 'Hello P of age unset!'

def test_unset_types_name_with_underscore():
    if False:
        i = 10
        return i + 15

    @strawberry.type
    class Query:
        hello: str = 'Hello'

    @strawberry.input
    class InputExample:
        first_name: str
        age: typing.Optional[str] = UNSET

    @strawberry.type
    class Mutation:

        @strawberry.mutation
        def say(self, first_name: typing.Optional[str]=UNSET) -> str:
            if False:
                while True:
                    i = 10
            if first_name is UNSET:
                return 'Name is unset'
            if first_name == '':
                return 'Hello Empty!'
            return f'Hello {first_name}!'

        @strawberry.mutation
        def say_age(self, input: InputExample) -> str:
            if False:
                return 10
            age = 'unset' if input.age is UNSET else input.age
            age = 'empty' if age == '' else age
            return f'Hello {input.first_name} of age {age}!'
    schema = strawberry.Schema(query=Query, mutation=Mutation)
    query = 'mutation {\n        one: say\n        two: say(firstName: "Patrick")\n        three: say(firstName: "")\n        empty: sayAge(input: { firstName: "Patrick", age: "" })\n        null: sayAge(input: { firstName: "Patrick", age: null })\n        sayAge(input: { firstName: "Patrick" })\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data['one'] == 'Name is unset'
    assert result.data['two'] == 'Hello Patrick!'
    assert result.data['three'] == 'Hello Empty!'
    assert result.data['empty'] == 'Hello Patrick of age empty!'
    assert result.data['null'] == 'Hello Patrick of age None!'
    assert result.data['sayAge'] == 'Hello Patrick of age unset!'

def test_unset_types_stringify_empty():
    if False:
        while True:
            i = 10

    @strawberry.type
    class Query:
        hello: str = 'Hello'

    @strawberry.type
    class Mutation:

        @strawberry.mutation
        def say(self, first_name: typing.Optional[str]=UNSET) -> str:
            if False:
                print('Hello World!')
            return f'Hello {first_name}!'
    schema = strawberry.Schema(query=Query, mutation=Mutation)
    query = 'mutation {\n        say\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data['say'] == 'Hello !'
    query = 'mutation {\n        say(firstName: null)\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data['say'] == 'Hello None!'

def test_converting_to_dict_with_unset():
    if False:
        return 10

    @strawberry.type
    class Query:
        hello: str = 'Hello'

    @strawberry.input
    class Input:
        name: typing.Optional[str] = UNSET

    @strawberry.type
    class Mutation:

        @strawberry.mutation
        def say(self, input: Input) -> str:
            if False:
                print('Hello World!')
            data = dataclasses.asdict(input)
            if data['name'] is UNSET:
                return 'Hello 🤨'
            return f"Hello {data['name']}!"
    schema = strawberry.Schema(query=Query, mutation=Mutation)
    query = 'mutation {\n        say(input: {})\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data['say'] == 'Hello 🤨'

def test_mutation_deprecation_reason():
    if False:
        i = 10
        return i + 15

    @strawberry.type
    class Query:
        hello: str = 'world'

    @strawberry.type
    class Mutation:

        @strawberry.mutation(deprecation_reason='Your reason')
        def say(self, name: str) -> str:
            if False:
                print('Hello World!')
            return f'Hello {name}!'
    schema = strawberry.Schema(query=Query, mutation=Mutation)
    assert str(schema) == dedent('        type Mutation {\n          say(name: String!): String! @deprecated(reason: "Your reason")\n        }\n\n        type Query {\n          hello: String!\n        }')