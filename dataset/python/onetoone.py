from typing import Any

from django.db import models
from django.db.models import OneToOneField

__all__ = ("OneToOneCascadeDeletes",)


class OneToOneCascadeDeletes(OneToOneField):
    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("on_delete", models.CASCADE)
        super().__init__(*args, **kwargs)
