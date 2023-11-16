from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, TypeVar
from typing_extensions import Protocol
from pydantic import BaseModel
if TYPE_CHECKING:
    from strawberry.types.types import StrawberryObjectDefinition
PydanticModel = TypeVar('PydanticModel', bound=BaseModel)

class StrawberryTypeFromPydantic(Protocol[PydanticModel]):
    """This class does not exist in runtime.
    It only makes the methods below visible for IDEs"""

    def __init__(self, **kwargs: Any):
        if False:
            return 10
        ...

    @staticmethod
    def from_pydantic(instance: PydanticModel, extra: Optional[Dict[str, Any]]=None) -> StrawberryTypeFromPydantic[PydanticModel]:
        if False:
            print('Hello World!')
        ...

    def to_pydantic(self, **kwargs: Any) -> PydanticModel:
        if False:
            i = 10
            return i + 15
        ...

    @property
    def __strawberry_definition__(self) -> StrawberryObjectDefinition:
        if False:
            while True:
                i = 10
        ...

    @property
    def _pydantic_type(self) -> Type[PydanticModel]:
        if False:
            for i in range(10):
                print('nop')
        ...