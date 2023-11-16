from typing import List
import strawberry

def a_resolver() -> List['AObject']:
    if False:
        while True:
            i = 10
    return []

@strawberry.type
class ABase:
    a_name: str

@strawberry.type
class AObject(ABase):
    a_age: int

    @strawberry.field
    def a_is_of_full_age(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.a_age >= 18