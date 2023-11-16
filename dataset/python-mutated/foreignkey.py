from __future__ import annotations
from typing import Any
from django.db import models
from django.db.backends.base.base import BaseDatabaseWrapper
from django.db.models import ForeignKey
__all__ = ('FlexibleForeignKey',)

class FlexibleForeignKey(ForeignKey):

    def __init__(self, *args: Any, **kwargs: Any):
        if False:
            for i in range(10):
                print('nop')
        kwargs.setdefault('on_delete', models.CASCADE)
        super().__init__(*args, **kwargs)

    def db_type(self, connection: BaseDatabaseWrapper) -> str | None:
        if False:
            while True:
                i = 10
        rel_field = self.target_field
        if hasattr(rel_field, 'get_related_db_type'):
            return rel_field.get_related_db_type(connection)
        return super().db_type(connection)