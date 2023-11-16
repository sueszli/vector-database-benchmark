import datetime
import random
from datetime import date
from typing import List, Type, cast
import pytest
from asgiref.sync import async_to_sync
from pytest_codspeed.plugin import BenchmarkFixture
import strawberry
from strawberry.scalars import ID

@pytest.mark.benchmark
def test_execute(benchmark: BenchmarkFixture):
    if False:
        i = 10
        return i + 15
    birthday = datetime.datetime.now()
    pets = ('cat', 'shark', 'dog', 'lama')

    @strawberry.type
    class Pet:
        id: int
        name: str

    @strawberry.type
    class Patron:
        id: int
        name: str
        age: int
        birthday: date
        tags: List[str]

        @strawberry.field
        def pets(self) -> List[Pet]:
            if False:
                return 10
            return [Pet(id=i, name=random.choice(pets)) for i in range(5)]

    @strawberry.type
    class Query:

        @strawberry.field
        def patrons(self) -> List[Patron]:
            if False:
                return 10
            return [Patron(id=i, name='Patrick', age=100, birthday=birthday, tags=['go', 'ajax']) for i in range(1000)]
    schema = strawberry.Schema(query=Query)
    query = '\n        query something{\n          patrons {\n            id\n            name\n            age\n            birthday\n            tags\n            pets {\n                id\n                name\n            }\n          }\n        }\n    '
    benchmark(async_to_sync(schema.execute), query)

@pytest.mark.parametrize('ntypes', [2 ** k for k in range(0, 13, 4)])
def test_interface_performance(benchmark: BenchmarkFixture, ntypes: int):
    if False:
        print('Hello World!')

    @strawberry.interface
    class Item:
        id: ID
    CONCRETE_TYPES: List[Type[Item]] = [strawberry.type(type(f'Item{i}', (Item,), {})) for i in range(ntypes)]

    @strawberry.type
    class Query:
        items: List[Item]
    schema = strawberry.Schema(query=Query, types=CONCRETE_TYPES)
    query = 'query { items { id } }'
    benchmark(async_to_sync(schema.execute), query, root_value=Query(items=[CONCRETE_TYPES[i % ntypes](id=cast(ID, i)) for i in range(1000)]))