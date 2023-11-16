from __future__ import annotations
from dataclasses import dataclass
from uuid import UUID
from litestar import Litestar, post
from litestar.dto import DataclassDTO, DTOConfig

@dataclass
class Person:
    id: UUID
    name: str
    age: int

class WriteDTO(DataclassDTO[Person]):
    """Don't allow client to set the id."""
    config = DTOConfig(exclude={'id'})

@post('/person', dto=WriteDTO, return_dto=None, sync_to_thread=False)
def create_person(data: Person) -> Person:
    if False:
        while True:
            i = 10
    'Create a person.'
    return data
app = Litestar(route_handlers=[create_person])