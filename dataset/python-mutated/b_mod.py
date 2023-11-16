from typing import List
import strawberry

def b_resolver() -> List['BObject']:
    if False:
        return 10
    return []

@strawberry.type
class BBase:
    b_name: str = strawberry.field()

@strawberry.type
class BObject(BBase):
    b_age: int = strawberry.field()

    @strawberry.field
    def b_is_of_full_age(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.b_age >= 18