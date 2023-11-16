from __future__ import annotations
from dataclasses import dataclass
from litestar import Litestar, post
from litestar.dto import DataclassDTO, DTOConfig, DTOData

@dataclass
class Address:
    id: int
    street: str

@dataclass
class Person:
    id: int
    name: str
    age: int
    address: Address

class ReadDTO(DataclassDTO[Person]):
    config = DTOConfig()

class WriteDTO(DataclassDTO[Person]):
    config = DTOConfig(exclude={'id', 'address.id'})

@post('/person', dto=WriteDTO, return_dto=ReadDTO, sync_to_thread=False)
def create_person(data: DTOData[Person]) -> Person:
    if False:
        return 10
    return data.create_instance(id=1, address__id=2)
app = Litestar(route_handlers=[create_person])