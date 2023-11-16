from typing import List, Optional, Tuple
from polyfactory.factories.pydantic_factory import ModelFactory
from pydantic import BaseModel
from litestar import Litestar, get
from litestar.pagination import AbstractSyncCursorPaginator, CursorPagination

class Person(BaseModel):
    id: str
    name: str

class PersonFactory(ModelFactory[Person]):
    __model__ = Person

class PersonCursorPaginator(AbstractSyncCursorPaginator[str, Person]):

    def __init__(self) -> None:
        if False:
            return 10
        self.data = PersonFactory.batch(50)

    def get_items(self, cursor: Optional[str], results_per_page: int) -> Tuple[List[Person], Optional[str]]:
        if False:
            for i in range(10):
                print('nop')
        results = self.data[:results_per_page]
        return (results, results[-1].id)
paginator = PersonCursorPaginator()

@get('/people', sync_to_thread=False)
def people_handler(cursor: Optional[str], results_per_page: int) -> CursorPagination[str, Person]:
    if False:
        return 10
    return paginator(cursor=cursor, results_per_page=results_per_page)
app = Litestar(route_handlers=[people_handler])