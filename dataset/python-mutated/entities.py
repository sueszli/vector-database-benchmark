"""Defines base entities used for providing lineage information."""
from __future__ import annotations
from typing import Any, ClassVar
import attr

@attr.s(auto_attribs=True)
class File:
    """File entity. Refers to a file."""
    template_fields: ClassVar = ('url',)
    url: str = attr.ib()
    type_hint: str | None = None

@attr.s(auto_attribs=True, kw_only=True)
class User:
    """User entity. Identifies a user."""
    email: str = attr.ib()
    first_name: str | None = None
    last_name: str | None = None
    template_fields: ClassVar = ('email', 'first_name', 'last_name')

@attr.s(auto_attribs=True, kw_only=True)
class Tag:
    """Tag or classification entity."""
    tag_name: str = attr.ib()
    template_fields: ClassVar = ('tag_name',)

@attr.s(auto_attribs=True, kw_only=True)
class Column:
    """Column of a Table."""
    name: str = attr.ib()
    description: str | None = None
    data_type: str = attr.ib()
    tags: list[Tag] = []
    template_fields: ClassVar = ('name', 'description', 'data_type', 'tags')

def default_if_none(arg: bool | None) -> bool:
    if False:
        while True:
            i = 10
    'Get default value when None.'
    return arg or False

@attr.s(auto_attribs=True, kw_only=True)
class Table:
    """Table entity."""
    database: str = attr.ib()
    cluster: str = attr.ib()
    name: str = attr.ib()
    tags: list[Tag] = []
    description: str | None = None
    columns: list[Column] = []
    owners: list[User] = []
    extra: dict[str, Any] = {}
    type_hint: str | None = None
    template_fields: ClassVar = ('database', 'cluster', 'name', 'tags', 'description', 'columns', 'owners', 'extra')