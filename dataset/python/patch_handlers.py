from __future__ import annotations

from dataclasses import dataclass

from litestar import Litestar, patch
from litestar.dto import DataclassDTO, DTOConfig, DTOData


@dataclass
class Person:
    name: str
    age: int
    email: str
    id: int


class ReadDTO(DataclassDTO[Person]):
    config = DTOConfig(exclude={"email"})


class PatchDTO(DataclassDTO[Person]):
    config = DTOConfig(exclude={"id"}, partial=True)


@patch("/person/{person_id:int}", dto=PatchDTO, return_dto=ReadDTO, sync_to_thread=False)
def update_person(person_id: int, data: DTOData[Person]) -> Person:
    # Usually the Person would be retrieved from a database
    person = Person(id=person_id, name="John", age=50, email="email_of_john@example.com")
    return data.update_instance(person)


app = Litestar(route_handlers=[update_person])
