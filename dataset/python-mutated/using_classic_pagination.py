from typing import List
from polyfactory.factories.pydantic_factory import ModelFactory
from pydantic import BaseModel
from litestar import Litestar, get
from litestar.pagination import AbstractSyncClassicPaginator, ClassicPagination

class Person(BaseModel):
    id: str
    name: str

class PersonFactory(ModelFactory[Person]):
    __model__ = Person

class PersonClassicPaginator(AbstractSyncClassicPaginator[Person]):

    def __init__(self) -> None:
        if False:
            return 10
        self.data = PersonFactory.batch(50)

    def get_total(self, page_size: int) -> int:
        if False:
            return 10
        return round(len(self.data) / page_size)

    def get_items(self, page_size: int, current_page: int) -> List[Person]:
        if False:
            while True:
                i = 10
        return [self.data[i:i + page_size] for i in range(0, len(self.data), page_size)][current_page - 1]
paginator = PersonClassicPaginator()

@get('/people', sync_to_thread=False)
def people_handler(page_size: int, current_page: int) -> ClassicPagination[Person]:
    if False:
        for i in range(10):
            print('nop')
    return paginator(page_size=page_size, current_page=current_page)
app = Litestar(route_handlers=[people_handler])