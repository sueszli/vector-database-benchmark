from typing import Optional
import pytest
from sqlalchemy import Column, Integer, String
from sqlmodel import Field, SQLModel

def test_sa_column_takes_precedence() -> None:
    if False:
        return 10

    class Item(SQLModel, table=True):
        id: Optional[int] = Field(default=None, sa_column=Column(String, primary_key=True, nullable=False))
    assert Item.id.nullable is False
    assert isinstance(Item.id.type, String)

def test_sa_column_no_sa_args() -> None:
    if False:
        print('Hello World!')
    with pytest.raises(RuntimeError):

        class Item(SQLModel, table=True):
            id: Optional[int] = Field(default=None, sa_column_args=[Integer], sa_column=Column(Integer, primary_key=True))

def test_sa_column_no_sa_kargs() -> None:
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(RuntimeError):

        class Item(SQLModel, table=True):
            id: Optional[int] = Field(default=None, sa_column_kwargs={'primary_key': True}, sa_column=Column(Integer, primary_key=True))

def test_sa_column_no_type() -> None:
    if False:
        i = 10
        return i + 15
    with pytest.raises(RuntimeError):

        class Item(SQLModel, table=True):
            id: Optional[int] = Field(default=None, sa_type=Integer, sa_column=Column(Integer, primary_key=True))

def test_sa_column_no_primary_key() -> None:
    if False:
        i = 10
        return i + 15
    with pytest.raises(RuntimeError):

        class Item(SQLModel, table=True):
            id: Optional[int] = Field(default=None, primary_key=True, sa_column=Column(Integer, primary_key=True))

def test_sa_column_no_nullable() -> None:
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(RuntimeError):

        class Item(SQLModel, table=True):
            id: Optional[int] = Field(default=None, nullable=True, sa_column=Column(Integer, primary_key=True))

def test_sa_column_no_foreign_key() -> None:
    if False:
        i = 10
        return i + 15
    with pytest.raises(RuntimeError):

        class Team(SQLModel, table=True):
            id: Optional[int] = Field(default=None, primary_key=True)
            name: str

        class Hero(SQLModel, table=True):
            id: Optional[int] = Field(default=None, primary_key=True)
            team_id: Optional[int] = Field(default=None, foreign_key='team.id', sa_column=Column(Integer, primary_key=True))

def test_sa_column_no_unique() -> None:
    if False:
        i = 10
        return i + 15
    with pytest.raises(RuntimeError):

        class Item(SQLModel, table=True):
            id: Optional[int] = Field(default=None, unique=True, sa_column=Column(Integer, primary_key=True))

def test_sa_column_no_index() -> None:
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(RuntimeError):

        class Item(SQLModel, table=True):
            id: Optional[int] = Field(default=None, index=True, sa_column=Column(Integer, primary_key=True))