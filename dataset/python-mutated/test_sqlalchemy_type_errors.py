from typing import Any, Dict, List, Optional, Union
import pytest
from sqlmodel import Field, SQLModel

def test_type_list_breaks() -> None:
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError):

        class Hero(SQLModel, table=True):
            id: Optional[int] = Field(default=None, primary_key=True)
            tags: List[str]

def test_type_dict_breaks() -> None:
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):

        class Hero(SQLModel, table=True):
            id: Optional[int] = Field(default=None, primary_key=True)
            tags: Dict[str, Any]

def test_type_union_breaks() -> None:
    if False:
        return 10
    with pytest.raises(ValueError):

        class Hero(SQLModel, table=True):
            id: Optional[int] = Field(default=None, primary_key=True)
            tags: Union[int, str]