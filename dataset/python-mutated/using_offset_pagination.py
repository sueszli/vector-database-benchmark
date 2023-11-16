from itertools import islice
from typing import List
from polyfactory.factories.pydantic_factory import ModelFactory
from pydantic import BaseModel
from litestar import Litestar, get
from litestar.pagination import AbstractSyncOffsetPaginator, OffsetPagination

class Person(BaseModel):
    id: str
    name: str

class PersonFactory(ModelFactory[Person]):
    __model__ = Person

class PersonOffsetPaginator(AbstractSyncOffsetPaginator[Person]):

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self.data = PersonFactory.batch(50)

    def get_total(self) -> int:
        if False:
            i = 10
            return i + 15
        return len(self.data)

    def get_items(self, limit: int, offset: int) -> List[Person]:
        if False:
            return 10
        return list(islice(islice(self.data, offset, None), limit))
paginator = PersonOffsetPaginator()

@get('/people', sync_to_thread=False)
def people_handler(limit: int, offset: int) -> OffsetPagination[Person]:
    if False:
        while True:
            i = 10
    return paginator(limit=limit, offset=offset)
app = Litestar(route_handlers=[people_handler])